import torch
import torch.nn as nn

'''
    these convs are designed to not change the size of image
'''
from droppath import ScheduleDropPath

from utils import calc_padding_ori,calc_padding


class BasicConv2d(nn.Sequential):
    def __init__(self,in_channels:int,out_channels:int,size:int,kernel_size:int,stride:int=1):
        super(BasicConv2d,self).__init__()
        # self.relu = nn.ReLU()
        self.relu = nn.ReLU6()
        self.conv=nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=calc_padding(size,kernel_size,stride),
                            bias=False)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001, momentum=0.1)



class _SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(_SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
                                        kernel_size=dw_kernel,
                                        stride=dw_stride,
                                        padding=dw_padding,
                                        bias=bias,
                                        groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)


class SeparableConv2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, size, kernel_size, stride=1, padding=None, bias=False):
        super(SeparableConv2d, self).__init__()
        if padding==None:
            padding=calc_padding(size,kernel_size,stride)

        # self.relu = nn.ReLU()
        self.relu = nn.ReLU6()
        self.separable_1 = _SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        # self.relu1 = nn.ReLU()
        self.relu1 = nn.ReLU6()
        self.separable_2 = _SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class DilatedConv2d(nn.Sequential):
    def __init__(self,in_channels:int,out_channels:int,size:int,kernel_size:int,stride:int=1,dilation:int=2):
        super(DilatedConv2d,self).__init__()
        # self.relu = nn.ReLU()
        self.relu = nn.ReLU6()
        real_kernel_size=dilation*(kernel_size-1)+1
        self.conv=nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=calc_padding(size,real_kernel_size,stride),
                            dilation=dilation,
                            bias=False)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001, momentum=0.1)


class BasicPolling2d(nn.Sequential):
    def __init__(self, in_channels: int, kernel_size: int,type='max'):
        super(BasicPolling2d,self).__init__()
        # self.remap = BasicConv2d(in_channels, out_channels, kernel_size=1)
        # self.relu = nn.ReLU()
        self.relu = nn.ReLU6()
        padding=(kernel_size-1)//2 # =calc_padding(size,kernel_size,1)
        if(type=='max'):
            self.pooling=nn.MaxPool2d(kernel_size=kernel_size,stride=1, padding=padding)
        elif(type=='avg'):
            self.pooling=nn.AvgPool2d(kernel_size=kernel_size,stride=1, padding=padding)

        self.bn=nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)

class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU6()
        conv1_out_channels = out_channels//2
        conv2_out_channels = out_channels-conv1_out_channels
        self.conv1 = nn.Conv2d(
            in_channels, conv1_out_channels, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            in_channels, conv2_out_channels, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(x)
        a = self.conv1(x)
        b = self.conv2(x[:, :, 1:, 1:])
        out = torch.cat([a, b], dim=1)
        out = self.bn(out)
        return out

class SimpleReduce(nn.Sequential):
    def __init__(self, input_size: int, output_size: int, in_channels: int,out_channels:int,kernel_size=4):
        super().__init__()
        self.relu=nn.ReLU6()
        self.conv=nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=2,
                      padding=calc_padding_ori(input_size, output_size, kernel_size, 2),
                      bias=False)
        self.bn=nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        


class Conv1x1Reduce(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU6()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

class ScheduleConvDropWarp(nn.Sequential):
    def __init__(self,op,scheduler):
        super().__init__()
        self.op=op
        self.scheduler=scheduler



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


