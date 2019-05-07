from cnn.CNN import CNN, ResBlock
from program.cnn.cell_elem import SeparableConv2dx2

from torchsummary import summary
import torch.nn as nn

if __name__ == "__main__":
    config_list={
        2: [(0, 1)],
        3: [(0, 1)],
        4: [(0, 1)],
        5: [(1, 1)],
        6: [(1, 1)],
        7: [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
    }
    cell_config_list = {'normal_cell': config_list}


    # cnn = CNN(cell_config_list,10)
    cnn=ResBlock(config_list,2,32)
    for param in cnn.modules():
        print(param)
    summary(cnn,(32,28,28))


    # net=nn.Sequential(
    #     nn.ReLU(),
    #     nn.Conv2d(3,64,3,1,1),
    #     nn.BatchNorm2d(64,eps=0.001, momentum=0.1, affine=True)
    # )
    #
    # summary(net,(3,28,28))
    # print(cnn.shape)
