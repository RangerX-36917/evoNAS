import torch

import torch.nn as nn
from collections import OrderedDict
from cell_elem import *
from utils import *
from cell import *
'''
    cell config description
    0,1: input node
    2~n-2: hidden state node
    n-1: output node
'''


class NASNetCIFAR(nn.Module):
    def __init__(self, cell_config: dict, stem_channels:int,cell_base_channels: int, num_normal_cells: int, image_size: int, num_classes: int):
        super(NASNetCIFAR, self).__init__()
        '''
            used for calc freature_map_num
            branch_num: the amount of nodes pointing to output-node
        '''
        self.num_classes = num_classes
        normal_cell_config = cell_config['normal_cell']
        reduce_cell_config=None

        _image_size = image_size
        _channels=cell_base_channels
        self.cellstem=CellStem0(3,stem_channels)
        self.normal_layer0=FirstCellStack(
            config=normal_cell_config,
            image_size=_image_size,
            num_normal_cells=num_normal_cells,
            in_channels=stem_channels,
            conv_channels=_channels
        )
        self.reduction_layer0 = ReduceCell()


        _image_size//=2
        _channels*=2
        out_channels0,_=get_stack_out_channels(self.normal_layer0)
        self.normal_layer1 = NormalCellStack(
            config=normal_cell_config,
            image_size=_image_size,
            num_normal_cells=num_normal_cells,
            in_channels=out_channels0,
            in_prev_channels=out_channels0,
            conv_channels=_channels
            )
        self.reduction_layer1 = ReduceCell()



        _image_size //= 2
        _channels*=2
        out_channels1,_=get_stack_out_channels(self.normal_layer1)
        self.normal_layer2 = NormalCellStack(
            config=normal_cell_config,
            image_size=_image_size,
            num_normal_cells=num_normal_cells,
            in_channels=out_channels1,
            in_prev_channels=out_channels1,
            conv_channels=_channels
            )
        self.reduction_layer2 = ReduceCell()
        _image_size //= 2

        out_channels2,_=get_stack_out_channels(self.normal_layer2)
        self.gap_layer = nn.AdaptiveAvgPool2d(1)

        # self.softmax_layer = nn.Sequential(OrderedDict([
        #         ('fc1',nn.Linear(in_features=out_channels2, out_features=100)),
        #         ('fc2',nn.Linear(in_features=100, out_features=num_classes))
        #     ]))

        self.softmax_layer = nn.Sequential(OrderedDict([
                ('dropout',nn.Dropout(0.2)),
                ('fc1',nn.Linear(in_features=out_channels2, out_features=num_classes))
            ]))
    

    def forward(self, images):
        x = self.cellstem(images)
        x,x_p=self.normal_layer0(x)
        x_p=x
        x=self.reduction_layer0(x,None)

        x,x_p=self.normal_layer1(x,x_p)
        x_p=x
        x=self.reduction_layer1(x,None)

        x,x_p=self.normal_layer2(x,x_p)

        x=self.gap_layer(x)

        x = x.view(x.size(0), -1)

        logits=self.softmax_layer(x)

        return logits


