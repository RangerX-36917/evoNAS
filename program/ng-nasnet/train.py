import time

import torch
import torch.nn as nn
import torch.optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


import random
import numpy as np
from utils import *
from load_dataset import *

from nasnet import NASNetCIFAR
from label_smoothing import CrossEntropyLossMOD
from lr_schedule import DampedCosineAnnealingWarmRestarts

import argparse
import json

from apex import amp
import apex.parallel

import torch.backends.cudnn as cudnn


def print_local(*text, **args):
    if local_rank == 0:
        print(*text, **args)


def train(train_loader, model, criterion,  optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # torch.autograd.set_detect_anomaly(True)

    model.train()

    begin_time = time.time()

    for step, data in enumerate(train_loader):

        data = tuple(t.cuda() for t in data)
        images, labels = data
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        prec1, = accuracy(output.detach(), labels, topk=(1,))

        reduced_loss = reduce_tensor(loss.detach())
        reduced_prec1 = reduce_tensor(prec1)

        losses.update(reduced_loss.item(), images.shape[0])
        top1.update(reduced_prec1.item(), images.shape[0])

        torch.cuda.synchronize()

        batch_time.update(time.time()-begin_time)

        if step % 10 == 0:
            print_local('Train: epoch:{:>4}: iter:{:>4} avg_batch_time: {:.3f} s loss:{:.4f} avg_loss:{:.4f} acc:{:.3f} avg_acc={:.3f} '.format(
                epoch, step, batch_time.avg, losses.val, losses.avg, top1.val, top1.avg))

        begin_time = time.time()

    return top1


def evaluate(val_loader, model, criterion, training=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    with torch.no_grad():
        begin_time = time.time()
        for step, data in enumerate(val_loader):
            data = tuple(t.cuda() for t in data)
            images, labels = data

            output = model(images)
            loss = criterion(output, labels)
            prec1, = accuracy(output.detach(), labels, topk=(1,))

            reduced_loss = reduce_tensor(loss.detach())
            reduced_prec1 = reduce_tensor(prec1)

            losses.update(reduced_loss.item(), images.shape[0])
            top1.update(reduced_prec1.item(), images.shape[0])

            batch_time.update(time.time()-begin_time)

            begin_time = time.time()

            if not training:
                print_local('Val  : epoch:{:>4}: iter:{:>4} avg_batch_time: {:.3f} s loss:{:.4f} avg_loss:{:.4f} acc:{:.3f} avg_acc={:.3f}'.format(
                    0, step, batch_time.avg, losses.val, losses.avg, top1.val, top1.avg))

    return top1


def save_model():
    pass


def load_model():
    pass


def main():
    seed = train_config['seed']+local_rank
    random.seed(seed)
    np.random.seed(seed)

    train_batch_size = train_config['train_batch_size']
    val_batch_size = train_config['val_batch_size']
    cell_config_list = {'normal_cell': normal_cell_config}

    cudnn.benchmark = True  # cudnn auto-tunner

    device = torch.device('cuda', local_rank)
    n_gpu = torch.cuda.device_count()

    torch.cuda.set_device(device)
    torch.manual_seed(seed)

    dist.init_process_group(backend='nccl')
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if train_config['epoch'] <= train_config['dropout_schedule_steps']:
        steps = train_config['epoch']
    else:
        steps = train_config['dropout_schedule_steps']

    model = NASNetCIFAR(
        cell_config=cell_config_list,
        stem_channels=nasnet_config['num_stem_channels'],
        cell_base_channels=nasnet_config['cell_base_channels'],
        num_stack_cells=nasnet_config['num_stack_cells'],
        image_size=32,
        num_classes=10,
        start_dropblock_prob=train_config['start_dropblock_rate'],
        end_dropblock_prob=train_config['end_dropblock_rate'],
        start_droppath_prob=train_config['start_droppath_rate'],
        end_droppath_prob=train_config['end_droppath_rate'],
        dropfc_prob=train_config['dropfc_rate'],
        steps=steps,
    )

    model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda()

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=train_config['lr'],
    #     betas=(0.5, 0.999),
    #     weight_decay=1e-4
    # )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_config['lr'],
        momentum=0.9,
        weight_decay=1e-3,
        nesterov=True
    )

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level='O1',
                                      loss_scale=2**11
                                      )

    model = DDP(model,device_ids=[local_rank],output_device=local_rank)

    train_loader, val_loader, classes = load_dataset(
        train_config['data_path'], train_batch_size, val_batch_size)

    # schedule_lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=100,
    #     T_mult=2,
    #     eta_min=1e-4
    # )

    # schedule_lr = DampedCosineAnnealingWarmRestarts(
    #     optimizer,
    #     damping=0.5,
    #     T_0=50,
    #     T_mult=2,
    #     eta_min=5e-4
    # )

    # schedule_lr=torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     train_config['epoch']-train_config['start_lr_schedule_epoch'],
    #     eta_min=1e-4
    # )

    schedule_lr=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        150,
        eta_min=1e-3
    )
    # schedule_lr=torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     [100,200,300],
    #     gamma=0.5
    # )

    criterion = torch.nn.CrossEntropyLoss().cuda()
    # criterion=CrossEntropyLossMOD(10).cuda()

    for epoch in range(train_config['epoch']):
        train_loader.sampler.set_epoch(epoch)

        print_local('epoch {} start'.format(epoch))
        print_local('current lr: {}'.format(schedule_lr.get_lr()[0]))

        begin_time = time.time()

        if epoch >= train_config['start_dropout_schedule_epoch']:
            update_dropout_schedule(model)

        prec1_train = train(train_loader, model, criterion,  optimizer, epoch)

        prec1_val = evaluate(val_loader, model, criterion)

        if epoch % 50 == 49:
            prec1_train_val = evaluate(
                train_loader, model, criterion, training=True)

        if epoch >= train_config['start_lr_schedule_epoch']:
            schedule_lr.step()

        print_local('train acc: {:.3f}'.format(prec1_train.avg))
        print_local('val acc: {:.3f}'.format(prec1_val.avg))

        if epoch % 50 == 49:
            print_local('val trainset acc: {:.3f}'.format(prec1_train_val.avg))

        print_local('total time: {:.3f} s'.format(time.time()-begin_time))
        print_local('\n\n')


if __name__ == '__main__':
    global local_rank, train_config, nasnet_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config.json',
                        type=str, help="config file path")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    args = parser.parse_args()

    local_rank = args.local_rank
    config_path = args.config_path

    with open(config_path, mode='r', encoding='utf-8') as f:
        config = json.load(f)

    train_config = config['train_config']
    nasnet_config = config['nasnet_config']
    # normal_cell_config = {
    #     2: [(0, 1), (1, 7)],
    #     3: [(0, 3), (1, 1),(2, 1)],
    #     4: [(0, 1), (1, 1), (2, 6)],
    #     5: [(0, 7), (1, 1), (4, 7)],
    #     6: [(0, 1), (1, 7)],
    #     7: [(3, 1), (5, 1), (6, 1)]
    # }

    # # nasnet-A
    normal_cell_config = {
        2: [(1, 7), (1, 7)],
        3: [(0, 7), (1, 6)],
        4: [(0, 1), (1, 2)],
        5: [(0, 2), (1, 2)],
        6: [(0, 7), (1, 7)],
        7: [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
    }

    # resnet 3 conv
    # normal_cell_config={
    #     2:[(0,1),(1,1)],
    #     3:[(2,7)],
    #     4:[(1,1),(3,1)],
    #     5:[(4,7)],
    #     6:[(3,1),(5,1)],
    #     7:[(6,7)]
    # }

    # # densenet 3 conv
    # normal_cell_config={
    #     2:[(0,1),(1,1)],
    #     3:[(2,7)],
    #     4:[(0,1),(1,1),(3,1)],
    #     5:[(4,7)],
    #     6:[(0,1),(1,1),(3,1),(5,1)],
    #     7:[(6,7)]
    # }

    main()
