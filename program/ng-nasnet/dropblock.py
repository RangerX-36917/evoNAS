import torch
import torch.nn.functional as F
from torch import nn
from utils import LinearScheduler

r'''
    see "https://github.com/miguelvr/dropblock", we modified it.
'''


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x,x.shape[2:])

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x,feat_size):
        gamma=self.drop_prob / (self.block_size ** 2)
        h,w=feat_size
        gamma*=(h*w)/((h-self.block_size+1)*(w-self.block_size+1))
        return gamma





# class ScheduleDropBlock(nn.Module):
#     def __init__(self,block_size, start_dropout_rate, stop_dropout_rate, steps):
#         super(ScheduleDropBlock,self).__init__()
#         self.dropblock=DropBlock2D(start_dropout_rate)
#         self.iter_cnt = 0
#         self.drop_values = torch.linspace(start=start_dropout_rate, stop=stop_dropout_rate, num=steps)


#     def forward(self, x):
#         return self.dropblock(x)

#     def step(self):
#         if self.iter_cnt < len(self.drop_values):
#             self.dropblock.drop_prob = self.drop_values[self.iter_cnt]

#         self.iter_cnt += 1


#TODO: slow speed. use cache to pre-generate mask
class DropBlock2D_Channel(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D_Channel, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x,x.shape[2:])

            # sample mask
            mask = (torch.rand(x.shape) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask

        return block_mask

    def _compute_gamma(self, x,feat_size):
        gamma=self.drop_prob / (self.block_size ** 2)
        h,w=feat_size
        gamma*=(h*w)/((h-self.block_size+1)*(w-self.block_size+1))
        return gamma


class ScheduleDropBlock(nn.Module):
    def __init__(self,block_size, start_dropout_rate, stop_dropout_rate, steps,per_channel=False):
        super(ScheduleDropBlock,self).__init__()
        if per_channel:
            self.dropblock=DropBlock2D_Channel(start_dropout_rate,block_size)
        else:
            self.dropblock=DropBlock2D(start_dropout_rate,block_size)
        self.schduler=LinearScheduler(self.dropblock,start_dropout_rate,stop_dropout_rate,steps)
        self.start_dropout_rate=start_dropout_rate
        self.stop_dropout_rate=stop_dropout_rate

    def forward(self,x):
        return self.schduler(x)
