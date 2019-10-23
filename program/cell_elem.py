import torch.nn as nn

'''
    these convs are designed to not change the size of image
'''


def calc_padding_ori(input_size,output_size,kernel_size,stride):
    return (stride*(output_size-1)+kernel_size-input_size)//2

def calc_padding(size,kernel_size,stride):
    '''
    the case when input_size=ouput_size
    '''
    return ((stride-1)*(size-1)-1+kernel_size)//2


class BasicConv2d(nn.Sequential):
    def __init__(self,in_channels:int,out_channels:int,size:int,kernel_size:int,stride:int=1):
        super(BasicConv2d,self).__init__()
        self.relu = nn.ReLU()
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

        self.relu = nn.ReLU()
        self.separable_1 = _SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = _SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class DilatedConv2d(nn.Sequential):
    def __init__(self,in_channels:int,out_channels:int,size:int,kernel_size:int,stride:int=1,dilation:int=2):
        super(DilatedConv2d,self).__init__()
        self.relu = nn.ReLU()
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
        self.relu = nn.ReLU()
        padding=(kernel_size-1)//2 # =calc_padding(size,kernel_size,1)
        if(type=='max'):
            self.pooling=nn.MaxPool2d(kernel_size=kernel_size,stride=1, padding=padding)
        elif(type=='avg'):
            self.pooling=nn.AvgPool2d(kernel_size=kernel_size,stride=1, padding=padding)

        self.bn=nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)