import torch
import torch.nn as nn
from cell_elem import *
from utils import *


class CellStem0(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(CellStem0, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(
            out_channels, eps=0.001, momentum=0.1, affine=True)


class FirstCellStack(nn.Module):
    def __init__(self, config: dict, image_size: int, num_normal_cells: int,  in_channels: int, conv_channels: int):
        super(FirstCellStack, self).__init__()
        if num_normal_cells < 2:
            raise ValueError('cell_stem normal cell number should >=2')
        self.num_normal_cells = num_normal_cells
        self.image_size = image_size
        self.in_channels = in_channels
        self.conv_channels = conv_channels

        self.cells = self._build_cells(config)
        self.out_channels = get_stack_out_channels(self)

    def _build_cells(self, config):
        normal_cells = nn.ModuleList()
        _channels = self.in_channels
        _prev_channels = self.in_channels

        for i in range(self.num_normal_cells):
            cell = NormalCell(config,
                              size=self.image_size,
                              in_channels=_channels,
                              in_prev_channels=_prev_channels,
                              conv_channels=self.conv_channels)

            _prev_channels = _channels
            _channels = cell.out_channels
            normal_cells.append(cell)

        return normal_cells

    def forward(self, x):
        # TODO: to check whether direct assignment is fine.
        x_prev = x
        for i in range(self.num_normal_cells):
            x_new = self.cells[i](x, x_prev)
            x_prev = x
            x = x_new

        return x, x_prev


class NormalCellStack(nn.Module):
    def __init__(self, config: dict,  image_size: int, num_normal_cells: int, in_channels: int, in_prev_channels: int, conv_channels: int):
        super(NormalCellStack, self).__init__()
        self.num_normal_cells = num_normal_cells
        self.image_size = image_size
        self.in_channels = in_channels
        self.in_prev_channels = in_prev_channels
        self.conv_channels = conv_channels

        self.cells = self._build_cells(config)
        self.out_channels = self.cells[-1].out_channels


    def _build_cells(self, config):
        normal_cells = nn.ModuleList()
        _channels = self.in_channels
        _prev_channels = self.in_prev_channels

        
        for i in range(self.num_normal_cells):
            if i == 0:
                cell = NormalCell(config,
                                  size=self.image_size,
                                  in_channels=_channels,
                                  in_prev_channels=_prev_channels,
                                  conv_channels=self.conv_channels,
                                  first_cell_flag=True)

            else:
                cell = NormalCell(config,
                                  size=self.image_size,
                                  in_channels=_channels,
                                  in_prev_channels=_prev_channels,
                                  conv_channels=self.conv_channels)
            # print('in_c{},in_p_c{}'.format(_channels,_prev_channels))
            _prev_channels = _channels
            _channels = cell.out_channels
            normal_cells.append(cell)

        return normal_cells

    def forward(self, x, x_prev):
        for i in range(self.num_normal_cells):
            # print(self.cells[i])
            x_new = self.cells[i](x, x_prev)
            x_prev = x
            x = x_new
        
        return x, x_prev


class NormalCell(nn.Module):
    def __init__(self, config: dict, size: int, in_channels: int, in_prev_channels: int, conv_channels: int, first_cell_flag: bool = False):
        super(NormalCell, self).__init__()

        self.in_channels = in_channels
        self.in_prev_channels = in_prev_channels
        
        self.output_node_id = len(config) + 1
        self.out_channels = len(config[self.output_node_id])*conv_channels
        
        # Note: the first cell need to deal with the prev_input from reduce_cell input that has a different image size
        self.first_cell_flag = first_cell_flag

        # input reducing
        if first_cell_flag:
            # use 1x1 conv
            self.preproc0 = FactorizedReduce(in_prev_channels, conv_channels)
        else:
            self.preproc0 = Conv1x1Reduce(in_prev_channels, conv_channels)

        self.preproc1 = Conv1x1Reduce(in_channels, conv_channels)

        self.convs = nn.ModuleDict()  # store conv operations

        # decode the raw cell config list and build the cell

        # Note: since the architecture is a DAG, the data flow at an order: from small node to large node
        for dstnode_id in range(2, self.output_node_id + 1):
            dstnode = 'node'+str(dstnode_id)
            self.convs[dstnode] = nn.ModuleDict()
            for srcnode_id, op in config[dstnode_id]:
                
                conv, name = choose_conv_elem(
                    op, size, conv_channels, conv_channels)
                srcnode="{}->{}->{}".format(srcnode_id, name,dstnode_id)
                self.convs[dstnode][srcnode] = conv



    def forward(self, x, x_prev):
        hidden_state = [None for _ in range(self.output_node_id+1)]
        hidden_state[0] = self.preproc0(x_prev)
        hidden_state[1] = self.preproc1(x)


        for i in range(2, self.output_node_id):
            data = []
            dstnode='node'+str(i)
            if len(self.convs[dstnode]) > 0:  # check whether this node is used or not
                for srcnode, conv in self.convs[dstnode].items():
                    srcnode_id=int(srcnode.split('->')[0])
                    data.append(conv(hidden_state[srcnode_id]))

                hidden_state[i] = torch.stack(data, dim=0).sum(dim=0)

        data_concat = []
        for srcnode, conv in self.convs['node'+str(self.output_node_id)].items():
            srcnode_id=int(srcnode.split('->')[0])
            data_concat.append(conv(hidden_state[srcnode_id]))

        x = torch.cat(data_concat, dim=1)
        return x


class ReduceCell(nn.Module):
    def __init__(self):
        super(ReduceCell, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, x_prev):
        '''
            currently x_prev is dummy param
        '''
        x_new = self.pool(x)

        return x_new

# TODO: implement another FR method in nasnet


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU()
        conv1_out_channels = out_channels//2
        conv2_out_channels = out_channels-conv1_out_channels
        self.conv1 = nn.Conv2d(
            in_channels, conv1_out_channels, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            in_channels, conv2_out_channels, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        

    def forward(self, x):
        x = self.relu(x)
        a=self.conv1(x)
        b=self.conv2(x[:, :, 1:, 1:])
        out = torch.cat([a,b], dim=1)
        out = self.bn(out)
        return out


class Conv1x1Reduce(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)


# TODO: add 5x5 pool?
def choose_conv_elem(op: int, size: int = None, in_channels: int = None, out_channels=None):
    if not (op >= 1 and op <= 7):
        raise ValueError("OP can only be a number in [1,7].")
    if (op == 1):  # identity
        conv = nn.Identity()
        name = 'identity'
    if (op == 2):  # 3x3 average pooling
        conv = BasicPolling2d(in_channels, kernel_size=3, type='avg')
        name = 'avg_pool_3x3'
    if (op == 3):  # 3x3 max pooling
        conv = BasicPolling2d(in_channels, kernel_size=3, type='max')
        name = 'max_pool_3x3'
    if (op == 4):  # 1x1 convolution
        conv = BasicConv2d(in_channels, out_channels, size=size, kernel_size=1)
        name = 'conv_1x1'
    if (op == 5):  # 3x3 depthwise-separable conv
        conv = SeparableConv2d(in_channels, out_channels,
                               size=size, kernel_size=3)
        name = 'sep_conv_3x3'
    if (op == 6):  # 3x3 dilated convolution->a 5x5 "dilated" filter(d=2)
        conv = DilatedConv2d(in_channels, out_channels,
                             size=size, kernel_size=3, dilation=2)
        name = 'dil_conv_3x3'
    if (op == 7):  # 3x3 convolution
        conv = BasicConv2d(in_channels, out_channels, size=size, kernel_size=3)
        name = 'conv_3x3'

    return conv, name
