import torch

import torch.nn as nn
from collections import OrderedDict
from cell_elem import *
from utils import *
from cell import *
from dropblock import ScheduleDropBlock
from droppath import ScheduleDropPath
'''
    cell config description
    0,1: input node
    2~n-2: hidden state node
    n-1: output node
'''


class NASNetCIFAR(nn.Module):
    def __init__(self, 
    cell_config: dict, 
    stem_channels:int,
    cell_base_channels: int, 
    num_stack_cells: int, 
    image_size: int, 
    num_classes: int,
    start_dropblock_prob:float,
    end_dropblock_prob:float,
    start_droppath_prob:float,
    end_droppath_prob:float,
    dropfc_prob:float,
    steps:int):
        super(NASNetCIFAR, self).__init__()
        '''
            steps: the number of mini-batch iterations
        '''
        self.num_classes = num_classes
        normal_cell_config = cell_config['normal_cell']
        reduce_cell_config=None

        # self.scheduled_drop=ScheduleDropPath(start_droppath_prob,end_droppath_prob,steps)

        _image_size = image_size # 32x32
        _channels=cell_base_channels

       
        self.dropblock0=ScheduleDropBlock(9,start_dropblock_prob,end_dropblock_prob,steps)
        self.droppath=ScheduleDropPath(start_droppath_prob,end_droppath_prob,steps)

        use_droppath_flag= start_dropblock_prob!=0 and end_dropblock_prob!=0
        self.cellstem=CellStem0(3,stem_channels)
        self.normal_layer0=FirstCellStack(
            config=normal_cell_config,
            image_size=_image_size,
            num_stack_cells=num_stack_cells,
            in_channels=stem_channels,
            conv_channels=_channels,
            scheduled_drop=self.droppath if use_droppath_flag else self.dropblock0
        )
        self.reduction_layer0 = ReduceCell(_image_size,_image_size//2,self.normal_layer0.out_channels)
        # self.reduction_layer0=ReduceCell_Old()
        
        
        _image_size//=2 # 16x16
        _channels*=2
        out_channels0=self.normal_layer0.out_channels

        self.dropblock1=ScheduleDropBlock(7,start_dropblock_prob,end_dropblock_prob,steps)
        self.normal_layer1 = NormalCellStack(
            config=normal_cell_config,
            image_size=_image_size,
            num_stack_cells=num_stack_cells,
            in_channels=out_channels0,
            in_prev_channels=out_channels0,
            conv_channels=_channels,
            scheduled_drop=self.droppath if use_droppath_flag else self.dropblock1
            )
        self.reduction_layer1 = ReduceCell(_image_size,_image_size//2,self.normal_layer1.out_channels)
        # self.reduction_layer1=ReduceCell_Old()
        
        
        _image_size //= 2 # 8x8
        _channels*=2
        out_channels1=self.normal_layer1.out_channels
        self.dropblock2=ScheduleDropBlock(5,start_dropblock_prob,end_dropblock_prob,steps)
        self.normal_layer2 = NormalCellStack(
            config=normal_cell_config,
            image_size=_image_size,
            num_stack_cells=num_stack_cells,
            in_channels=out_channels1,
            in_prev_channels=out_channels1,
            conv_channels=_channels,
            scheduled_drop=self.droppath if use_droppath_flag else self.dropblock2
            )

        _image_size //= 2

        out_channels2=self.normal_layer2.out_channels
        self.gap_layer = nn.AdaptiveAvgPool2d(1)

        # self.softmax_layer = nn.Sequential(OrderedDict([
        #         ('fc1',nn.Linear(in_features=out_channels2, out_features=100)),
        #         ('fc2',nn.Linear(in_features=100, out_features=num_classes))
        #     ]))

        self.softmax_layer = nn.Sequential(OrderedDict([
                ('dropout1',nn.Dropout(dropfc_prob)),
                ('fc1',nn.Linear(in_features=out_channels2, out_features=100)),
                ('dropout2',nn.Dropout(dropfc_prob)),
                ('fc2',nn.Linear(in_features=100, out_features=num_classes))
            ]))

    

    def forward(self, images):
        x = self.cellstem(images)

        x,x_p=self.normal_layer0(x)
        x_new=self.reduction_layer0(x,x_p)
        x,x_p=x_new,x

        x=self.dropblock1(x)

        x,x_p=self.normal_layer1(x,x_p)
        x_new=self.reduction_layer1(x,x_p)
        x,x_p=x_new,x

        x=self.dropblock2(x)

        x,x_p=self.normal_layer2(x,x_p)

        x=self.gap_layer(x)

        x = x.view(x.size(0), -1)

        logits=self.softmax_layer(x)

        return logits


