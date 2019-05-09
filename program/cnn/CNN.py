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
    def __init__(self, cell_config_list: dict, class_num: int, image_size=32,N: int = 2):
        super(CNN, self).__init__()
        '''
            used for calc freature_map_num
            branch_num: the amount of nodes pointing to output-node
        '''
        normal_cell_conf = cell_config_list['normal_cell']

        self.class_num = class_num

        _size=image_size

        # TODO: use parameter representation (current only suitable for image with 3 channels)
        channel1 = 32
        self.normal_layer1 = ResBlock(normal_cell_conf, N, conv_channels=channel1, in_channels=3)
        self.reduction_layer1 = nn.MaxPool2d(kernel_size=2)
        _size//=2

        channel2 = 64
        self.normal_layer2 = ResBlock(normal_cell_conf, N, conv_channels=channel2, in_channels=channel1)
        self.reduction_layer2 = nn.MaxPool2d(kernel_size=2)
        _size //= 2

        channel3 = 128
        self.normal_layer3 = ResBlock(normal_cell_conf, N, conv_channels=channel3, in_channels=channel2)

        channel4 = 512
        self.conv1x1 = choose_conv_elem(4, in_channels=channel3, out_channels=channel4)


        # print(_size)
        self.gap_layer = nn.AvgPool2d(kernel_size=int(_size))

        self.fc1 = nn.Linear(in_features=channel4, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=class_num)
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # input(28,28)
        cnn_part = nn.Sequential(
            self.normal_layer1,  # -> (32,32,32)
            self.reduction_layer1,
            self.normal_layer2,  # -> (64,16,16)
            self.reduction_layer2,
            self.normal_layer3,  # -> (128,8,8)
            self.conv1x1,  # -> (512,8,8)
            self.gap_layer,  # -> (512,1,1)
        )

        softmax_part = nn.Sequential(
            self.fc1,
            self.fc2,
            self.softmax_layer
        )

        x = cnn_part(x)
        x = x.view(x.size(0), -1)
        x = softmax_part(x)
        # print('+++++++++++++++++++')
        return x


class ResBlock(nn.Module):
    def __init__(self, config_list: dict, N: int, conv_channels: int, in_channels: int):
        super(ResBlock, self).__init__()
        # _output_node_idx = len(config_list) + 1
        # intern_in_channels = len(config_list[_output_node_idx]) * channels

        self.normal_cells = []
        prev_cell = None
        for i in range(N):
            if i == 0:
                cell = Cell(config_list,
                            in_channels=in_channels,
                            in_skip_channels=in_channels,
                            conv_channels=conv_channels)
            else:
                cell = Cell(config_list,
                            in_channels=prev_cell.out_channels,
                            in_skip_channels=prev_cell.in_channels,
                            conv_channels=conv_channels)

            self.normal_cells.append(cell)
            self.add_module('cell{}'.format(i), cell)
            prev_cell = cell

        self.normal_cells_num = N

    def forward(self, x):
        # cell stack model
        x_skip = x

        # note: cell(x_in, x_skip)
        for i in range(self.normal_cells_num):
            if i == 0:
                x = self.normal_cells[i](x, x)
            else:
                x_tmp = x
                x = self.normal_cells[i](x, x_skip)
                x_skip = x_tmp
            # print('===============')

        return x


class Cell(nn.Module):
    def __init__(self, config_list: dict, in_channels: int, in_skip_channels: int, conv_channels: int,
                 output_cell_flag=False):
        super(Cell, self).__init__()

        self.in_channels = in_channels
        self.output_node_idx = len(config_list) + 1
        self.output_cell_flag = output_cell_flag
        self.convs = {}  # store conv operations

        # decode the raw cell config list and build the cell
        self.node_channels = {0: in_skip_channels, 1: in_channels}

        for dstnode in range(2, self.output_node_idx + 1):
            self.convs[dstnode] = []
            self.node_channels[dstnode] = 0
            # build net from # to dstnode
            for srcnode, opt in config_list[dstnode]:
                if opt >= 1 and opt <= 7:
                    # note: srcnode < dstnode => self.node_channels[srcnode] has been calculated
                    conv = choose_conv_elem(opt, self.node_channels[srcnode], conv_channels)
                    self.convs[dstnode].append((srcnode, conv))
                    self.add_module("conv_{}to{}".format(srcnode, dstnode), conv)
                    # print("conv_{}to{}".format(srcnode,dstnode),conv)

                    # update self.node_channels
                    if opt in (1, 2, 3):
                        # note: identity & pooling keep the original channel num
                        self.node_channels[dstnode] += self.node_channels[srcnode]
                    else:
                        self.node_channels[dstnode] += conv_channels

        if (not self.output_cell_flag):
            # use 1x1 conv
            self.output_conv = choose_conv_elem(4, self.node_channels[self.output_node_idx], conv_channels)
            self.out_channels = conv_channels
        else:
            self.out_channels = self.node_channels[self.output_node_idx]

    def forward(self, x_in, x_skip):
        # print(x_in.shape)
        # print(x_skip.shape)
        hidden_state = [None for _ in range(self.output_node_idx + 1)]
        hidden_state[0] = x_skip
        hidden_state[1] = x_in
        for i in range(2, self.output_node_idx + 1):
            data = []
            if (len(self.convs[i]) > 0):  # check whether this node is used or not
                for srcnode, conv in self.convs[i]:
                    data.append(conv(hidden_state[srcnode]))

                hidden_state[i] = torch.cat(data, 1)

        if (not self.output_cell_flag):
            x = self.output_conv(hidden_state[self.output_node_idx])
        else:
            x = hidden_state[self.output_node_idx]

        return x


def choose_conv_elem(opt: int, in_channels=None, out_channels=None):
    conv = None

    if (opt == 1):  # identity
        conv = nn.Identity()
    if (opt == 2):  # 3x3 average pooling
        conv = BasicPolling2d(in_channels, kernel_size=3, type='avg')
    if (opt == 3):  # 3x3 max pooling
        conv = BasicPolling2d(in_channels, kernel_size=3, type='max')
    if (opt == 4):  # 1x1 convolution
        conv = BasicConv2d(in_channels, out_channels, kernel_size=1)
    if (opt == 5):  # 3x3 depthwise-separable conv
        conv = SeparableConv2dx2(in_channels, out_channels, kernel_size=3)
    if (opt == 6):  # 3x3 dilated convolution->a 5x5 "dilated" filter(d=2)
        conv = DilatedConv2d(in_channels, out_channels, kernel_size=3, dilation=2)
    if (opt == 7):  # 3x3 convolution
        conv = BasicConv2d(in_channels, out_channels, kernel_size=3)

    return conv
