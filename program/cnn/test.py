from cnn.CNN import CNN, ResBlock, Cell, choose_conv_elem
from cnn.cell_elem import SeparableConv2dx2

from torchsummary import summary
import torch.nn as nn

from cnn.cell_elem import *

if __name__ == "__main__":
    config_list = {
        2: [(0, 1)],
        3: [(0, 6)],
        4: [(0, 1)],
        5: [(1, 4)],
        6: [(1, 1), (4, 1), (5, 1)],
        7: [(2, 1), (3, 1), (4, 2), (5, 1), (6, 1)]
    }
    # config_list = {2: [(0, 1), (1, 5)],
    #                3: [(0, 5), (1, 3)],
    #                4: [(0, 1), (1, 2), (2, 7), (3, 5)],
    #                5: [(0, 4), (3, 4), (4, 7)],
    #                6: [(0, 3), (1, 5), (2, 1), (3, 2), (4, 5)],
    #                7: [(5, 1), (6, 1)]}

    cell_config_list = {'normal_cell': config_list}

    # conv=choose_conv_elem(3,3,64)
    #
    # summary(conv,(3,28,28))

    cnn = CNN(cell_config_list, class_num=10)
    # cnn=ResBlock(config_list,2,channels=128,in_channels=32)
    # cnn=Cell(config_list,in_channels=32,conv_channels=128)
    # print(cnn)
    # summary(cnn,[(32,28,28),(32,28,28)])
    summary(cnn, (3, 32, 32))

    # net=nn.Sequential(
    #     nn.ReLU(),
    #     nn.Conv2d(3,64,3,1,1),
    #     nn.BatchNorm2d(64,eps=0.001, momentum=0.1, affine=True)
    # )
    #
    # summary(net,(3,28,28))
    # print(cnn.shape)
