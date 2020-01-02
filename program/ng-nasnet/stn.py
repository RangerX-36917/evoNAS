import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from utils import calc_padding


class _STNConv2d(nn.Sequential):
    def __init__(self,in_channels:int,out_channels:int,size:int,kernel_size:int,stride:int=1):
        super(_STNConv2d,self).__init__()
        # self.relu = nn.ReLU()
        self.relu = nn.ReLU6()
        self.conv=nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=calc_padding(size,kernel_size,stride),
                            bias=False)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001, momentum=0.1)


class _Conv1x1Reduce(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU6()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

class SpatialTransformer(nn.Module):
    def __init__(self,size, in_channels, channels):
        super(SpatialTransformer, self).__init__()

        # Spatial transformer localization-network
        # self.localization = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # ) # -> 10x3x3

        _out_size=4

        

        self.conv1x1_reduce=_Conv1x1Reduce(in_channels,channels)


        _channels=channels
        _size=size
        self.localization = nn.ModuleList() 
        for i in range(np.log2(size/_out_size)-1):
            self.localization[i]=_STNConv2d(_channels,_channels//2,_size,kernel_size=3)
            _channels//=2
            _size//=2


        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, x):
        # transform the input
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


