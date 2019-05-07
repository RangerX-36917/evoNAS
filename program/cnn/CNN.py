import torch

import torch.nn as nn

from program.cnn.cell_elem import BasicConv2d, SeparableConv2dx2, DilatedConv2d

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
        __output_node_idx = len(normal_cell_conf) + 1
        branch_num = len(normal_cell_conf[__output_node_idx])
        print('branch num:', branch_num)

        self.class_num = class_num

        channel1 = 32
        self.normal_layer1 = ResBlock(normal_cell_conf, N, channel1)  # -> (28,28,#)
        self.reduction_layer1 = nn.MaxPool2d(kernel_size=2)
        feature_map_num1 = channel1 * branch_num

        channel2 = 64
        self.normal_layer2 = ResBlock(normal_cell_conf, N, channel2)  # -> (14,14,#)
        self.reduction_layer2 = nn.MaxPool2d(kernel_size=2)
        feature_map_num2 = channel2 * branch_num

        channel3 = 128
        self.normal_layer3 = ResBlock(normal_cell_conf, N, channel3)  # -> (7,7,#)
        self.reduction_layer3 = nn.MaxPool2d(kernel_size=2)
        feature_map_num3 = channel3 * branch_num

        # TODO: use parameter representation (current only suitable for 28*28 image)
        self.gap_layer = nn.AvgPool2d(kernel_size=7)

        self.fc1 = nn.Linear(in_features=feature_map_num3, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=class_num)
        self.softmax_layer = nn.Softmax()

    def forward(self, x):
        # input(28,28)
        # cnn_part = nn.Sequential(
        #     self.normal_layer1,
        #     self.reduction_layer1,
        #     self.normal_layer2,
        #     self.reduction_layer2,
        #     self.normal_layer3,
        #     self.reduction_layer3,
        #     self.gap_layer,
        # )
        # x=cnn_part(x)

        x = self.normal_layer1(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, config_list: dict, N: int, in_channels: int, channels: int):
        super(ResBlock, self).__init__()
        self.normal_cells = [Cell(config_list, in_channels, channels) for _ in range(N)]

        # self.add_module()
        self.normal_cells_num = N

    def forward(self, x):
        x_skip = x
        for i in range(self.normal_cells_num):
            if i == 0:
                x = self.normal_cells[i](x,x)
            else:
                x_tmp = x
                x = self.normal_cells[i](x,x_skip)
                x_skip = x_tmp
        return x


class Cell(nn.Module):
    def __init__(self, config_list: dict, in_channels: int, conv_channels: int):
        super(Cell, self).__init__()
        self.output_node_idx = len(config_list) + 1
        self.config_list = {}

        # decode the raw cell config list
        for dstnode, srclist in config_list.items():
            self.config_list[dstnode] = []
            for srcnode, opt in srclist:
                if opt >= 1 or opt <= 7:
                    if srcnode == 0 or srcnode == 1:
                        self.config_list[dstnode].append(
                            (srcnode, self.choose_conv_elem(opt, in_channels, conv_channels)))
                    else:
                        self.config_list[dstnode].append(
                            (srcnode, self.choose_conv_elem(opt, conv_channels * len(srclist), conv_channels)))

    def choose_conv_elem(self, opt: int, in_channels=None, out_channels=None):
        conv = None
        if (opt == 1):  # identity
            conv = nn.Identity()
        if (opt == 2):  # 3x3 average pooling
            conv = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        if (opt == 3):  # 3x3 max pooling
            conv = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        if (opt == 4):  # 1x1 convolution
            conv = BasicConv2d(in_channels, out_channels, kernel_size=1)
        if (opt == 5):  # 3x3 depthwise-separable conv
            conv = SeparableConv2dx2(in_channels, out_channels, kernel_size=3)
        if (opt == 6):  # 3x3 dilated convolution->a 5x5 "dilated" filter(d=2)
            conv = DilatedConv2d(in_channels, out_channels, kernel_size=3, dilation=2)
        if (opt == 7):  # 3x3 convolution
            conv = BasicConv2d(in_channels, out_channels, kernel_size=3)

        return conv

    def forward(self, x_prev, x_skip):
        hidden_state = [x_skip, x_prev]
        for i in range(2, self.output_node_idx + 1):
            data = []
            for srcnode, conv in self.config_list[i]:
                data.append(conv(hidden_state[srcnode]))

            hidden_state.append(torch.cat(data, 1))

        return hidden_state[self.output_node_idx]
