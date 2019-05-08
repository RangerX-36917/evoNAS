import torch

import torch.nn as nn

from cnn.cell_elem import *

'''
    cell config description
    0,1: input node
    2~n-2: hidden state node
    n-1: output node
'''


class CNN(nn.Module):
    def __init__(self, cell_config_list: dict, class_num: int, N: int = 2):
        super(CNN, self).__init__()
        '''
            used for calc freature_map_num
            branch_num: the amount of nodes pointing to output-node
        '''
        normal_cell_conf = cell_config_list['normal_cell']
        _output_node_idx = len(normal_cell_conf) + 1
        branch_num = len(normal_cell_conf[_output_node_idx])
        # print('branch num:', branch_num)

        self.class_num = class_num

        # TODO: use parameter representation (current only suitable for image with 3 channels)
        channel1 = 32
        self.normal_layer1 = ResBlock(normal_cell_conf, N, channels=channel1, in_channels=3)
        self.reduction_layer1 = nn.MaxPool2d(kernel_size=2)
        feature_map_num1 = channel1 * branch_num

        channel2 = 64
        self.normal_layer2 = ResBlock(normal_cell_conf, N, channels=channel2, in_channels=channel1)
        self.reduction_layer2 = nn.MaxPool2d(kernel_size=2)
        feature_map_num2 = channel2 * branch_num

        channel3 = 128
        self.normal_layer3 = ResBlock(normal_cell_conf, N, channels=channel3, in_channels=channel2)

        feature_map_num3 = channel3 * branch_num

        # TODO: use parameter representation (current only suitable for 32*32 image)
        self.gap_layer = nn.AvgPool2d(kernel_size=8)

        self.fc1 = nn.Linear(in_features=channel3, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=class_num)
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # input(28,28)
        cnn_part = nn.Sequential(
            self.normal_layer1,  # -> (32,32,#)
            self.reduction_layer1,
            self.normal_layer2,  # -> (16,16,#)
            self.reduction_layer2,
            self.normal_layer3,  # -> (8,8,#)
            self.gap_layer,  # -> (1,1,#)
        )

        softmax_part = nn.Sequential(
            self.fc1,
            self.fc2,
            self.softmax_layer
        )

        x = cnn_part(x)
        x=x.view(x.size(0),-1)
        x=softmax_part(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, config_list: dict, N: int, channels: int, in_channels: int):
        super(ResBlock, self).__init__()
        # _output_node_idx = len(config_list) + 1
        # intern_in_channels = len(config_list[_output_node_idx]) * channels

        self.normal_cells = []
        for i in range(N):
            if i == 0:
                cell = Cell(config_list, in_channels, channels)
            else:
                cell = Cell(config_list, channels, channels)
            self.normal_cells.append(cell)
            self.add_module('cell{}'.format(i), cell)

        # self.add_module()
        self.normal_cells_num = N

        self.skip_remap_conv = choose_conv_elem(1, in_channels=in_channels, out_channels=channels)

    def forward(self, x):
        # cell stack model
        x_skip = self.skip_remap_conv(x)
        for i in range(self.normal_cells_num):
            if i == 0:
                x = self.normal_cells[i](x, x)
            else:
                # print("=======")
                x_tmp = x
                x = self.normal_cells[i](x, x_skip)
                x_skip = x_tmp

        return x


class Cell(nn.Module):
    def __init__(self, config_list: dict, in_channels: int, conv_channels: int, output_cell=False):
        super(Cell, self).__init__()
        self.output_node_idx = len(config_list) + 1
        self.output_cell = output_cell
        self.config_list = {}

        # decode the raw cell config list
        for dstnode, srclist in config_list.items():
            self.config_list[dstnode] = []
            for srcnode, opt in srclist:
                if opt >= 1 and opt <= 7:
                    if srcnode == 0 or srcnode == 1:
                        conv = choose_conv_elem(opt, in_channels, conv_channels)
                        self.config_list[dstnode].append((srcnode, conv))
                    else:
                        conv = choose_conv_elem(opt, conv_channels * len(config_list[srcnode]), conv_channels)
                        self.config_list[dstnode].append((srcnode, conv))

                    self.add_module("conv_{}to{}".format(srcnode, dstnode), conv)
                    # print("conv_{}to{}".format(srcnode,dstnode),conv)

        if (not self.output_cell):
            self.output_conv = choose_conv_elem(1, conv_channels * len(config_list[self.output_node_idx]),
                                                conv_channels)

    def forward(self, x_prev, x_skip):
        # print(x_prev.shape)
        # print(x_skip.shape)
        hidden_state = [None for _ in range(self.output_node_idx+1)]
        hidden_state[0]=x_skip
        hidden_state[1]=x_prev
        for i in range(2, self.output_node_idx + 1):
            data = []
            if(len(self.config_list[i])>0):
                for srcnode, conv in self.config_list[i]:
                    data.append(conv(hidden_state[srcnode]))

            if(len(data)>0):
                hidden_state[i]=torch.cat(data, 1)


        if (not self.output_cell):
            x = self.output_conv(hidden_state[self.output_node_idx])
        else:
            x = hidden_state[self.output_node_idx]

        return x


def choose_conv_elem(opt: int, in_channels=None, out_channels=None):
    conv = None
    if (opt == 1):  # identity -> # 1x1 convolution
        # conv = nn.Identity()
        conv = BasicConv2d(in_channels, out_channels, kernel_size=1)
    if (opt == 2):  # 3x3 average pooling
        conv = BasicPolling2d(in_channels,out_channels,kernel_size=3,type='avg')
    if (opt == 3):  # 3x3 max pooling
        conv = BasicPolling2d(in_channels,out_channels,kernel_size=3,type='max')
    if (opt == 4):  # 1x1 convolution
        conv = BasicConv2d(in_channels, out_channels, kernel_size=1)
    if (opt == 5):  # 3x3 depthwise-separable conv
        conv = SeparableConv2dx2(in_channels, out_channels, kernel_size=3)
    if (opt == 6):  # 3x3 dilated convolution->a 5x5 "dilated" filter(d=2)
        conv = DilatedConv2d(in_channels, out_channels, kernel_size=3, dilation=2)
    if (opt == 7):  # 3x3 convolution
        conv = BasicConv2d(in_channels, out_channels, kernel_size=3)

    return conv
