import torch.nn as nn

'''
    these convs are designed to not change the size of image
'''



class BasicConv2d(nn.Sequential):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int,stride:int=1):
        super(BasicConv2d,self).__init__()
        self.relu = nn.ReLU()
        self.conv=nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size-1)//2,
                            bias=False)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001, momentum=0.1)




class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)


class SeparableConv2dx2(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, bias=False):
        super(SeparableConv2dx2, self).__init__()
        if(padding==None):
            padding=(kernel_size-1)/2

        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class DilatedConv2d(nn.Sequential):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int,stride:int=1,dilation:int=2):
        super(DilatedConv2d,self).__init__()
        self.relu = nn.ReLU()
        self.conv=nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=((kernel_size-1)*(dilation-1)+kernel_size-1)//2,
                            dilation=dilation,
                            bias=False)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001, momentum=0.1)