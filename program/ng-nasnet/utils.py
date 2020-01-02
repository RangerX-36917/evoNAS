import torch
import torch.distributed as dist
from torch import nn
import math

# note: this equation selects the maximum padding it can get
def calc_padding_ori(input_size,output_size,kernel_size,stride):
    return math.ceil((stride*(output_size-1)+kernel_size-input_size)/2)

def calc_padding(size,kernel_size,stride):
    '''
    the case when input_size=ouput_size
    '''
    return calc_padding_ori(size,size,kernel_size,stride)


def get_stack_out_channels(cellstack):
    cells=cellstack.cells
    assert len(cells) >= 1
    out_channels = cells[-1].out_channels
    if len(cells) == 1:
        out_prev_channels = cells[-1].in_channels
    else:
        out_prev_channels = cells[-2].out_channels

    return out_channels,out_prev_channels





class LinearScheduler(nn.Module):
    def __init__(self, module, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.module = module
        self.iter = 0
        self.drop_values = torch.linspace(start=start_value, end=stop_value, steps=nr_steps)


    def forward(self, x):
        return self.module(x)

    def step(self):
        if self.iter < len(self.drop_values):
            self.module.drop_prob = self.drop_values[self.iter]

        self.iter += 1


def _check_and_update(m):
    if isinstance(m,LinearScheduler):
        m.step()


def update_dropout_schedule(module:nn.Module):    
    module.apply(_check_and_update)
        


def reduce_tensor(tensor,avg=True):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if avg:
        world_size=dist.get_world_size() 
        rt /= world_size
    
    return rt



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val=val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100/batch_size))
        # res.append(correct_k)
    return res


def correct_cnt(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        # res.append(correct_k.div_(batch_size))
        res.append(correct_k)
    return res
    


