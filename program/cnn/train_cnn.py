import time

import torch.nn
import torch
import torch.optim
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from cnn.CNN import CNN

LR = 0.001
DROPOUT = 0.2
EPOCH = 200
BATCH_SIZE = 50
DATASET_PATH = './dataset'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model: torch.nn.Module, trainloader, testloader):
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer=torch.nn.DataParallel(optimizer)
    loss_func = torch.nn.CrossEntropyLoss()
    # model.to(device)

    for epoch in range(EPOCH):
        start_time = time.time()

        running_loss = 0.0
        for step, data in enumerate(trainloader, 0):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            output = model(images)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 50 == 49:
                print('epoch: {0}, iter:{1} loss:{2:.4f}'.format(epoch, step, running_loss / 50))
                running_loss = 0
                pass
        print('epoch{}: '.format(epoch))
        evaluate(model, testloader)
        print('epoch {} finished, cost {:.3f} sec'.format(epoch, time.time() - start_time))
        print('=======================\n\n\n')


def evaluate(model: torch.nn.Module, testloader):
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # if total>100:
            #     break

    acc = correct / total * 100
    print('Accuracy of the network on the 10000 test images: {:.3f}'.format(acc))

    return acc


def load_dataset(path: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root=path, train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root=path, train=False,
                               download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def save_model():
    pass


def load_model():
    pass


if __name__ == '__main__':
    trainloader, testloader, classes = load_dataset(DATASET_PATH)
    '''
    config_list = {
        2: [(0, 1)],
        3: [(0, 6)],
        4: [(0, 1)],
        5: [(1, 4)],
        6: [(1, 1), (4, 1), (5, 1)],
        7: [(2, 1), (3, 1), (4, 3), (5, 1), (6, 1)]
    }

    config_list = {
        2: [(0, 4), (1, 6)],
        3: [(1, 7), (2, 2)],
        4: [(0, 4), (1, 3), (2, 1), ],
        5: [(1, 6), (2, 7), (3, 6), (4, 7)],
        6: [(0, 2), (1, 1), (2, 6), (3, 1), (4, 1), ],
        7: [(5, 1), (6, 1)]
    }
    '''

    config_list = {
        2: [(0, 2)],
        3: [(2, 4)],
        4: [(3, 4)],
        5: [(1, 6), (2,6)],
        6: [(0, 5)],
        7: [(4, 1), (5, 1), (6,1)]
    }
    cell_config_list = {'normal_cell': config_list}

    model = CNN(cell_config_list, class_num=len(classes),N=1)

    # train(model, trainloader)

    evaluate(model, testloader)
