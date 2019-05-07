from cnn.CNN import CNN, ResBlock, Cell
from program.cnn.cell_elem import SeparableConv2dx2

from torchsummary import summary
import torch.nn as nn

if __name__ == "__main__":
    config_list = {
        2: [(0, 1)],
        3: [(0, 6)],
        4: [(0, 1)],
        5: [(1, 4)],
        6: [(1, 1), (4, 1), (5, 1)],
        7: [(2, 1), (3, 1), (4, 3), (5, 1), (6, 1)]
    }


    cell_config_list = {'normal_cell': config_list}

    cnn = CNN(cell_config_list, class_num=10)
    # cnn=ResBlock(config_list,2,channels=128,in_channels=32)
    # cnn=Cell(config_list,in_channels=32,conv_channels=128)
    # print(cnn)
    # summary(cnn,[(32,28,28),(32,28,28)])
    summary(cnn, (3, 28, 28))

    # net=nn.Sequential(
    #     nn.ReLU(),
    #     nn.Conv2d(3,64,3,1,1),
    #     nn.BatchNorm2d(64,eps=0.001, momentum=0.1, affine=True)
    # )
    #
    # summary(net,(3,28,28))
    # print(cnn.shape)
