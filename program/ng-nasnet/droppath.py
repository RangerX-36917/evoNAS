import torch
from torch import nn
from utils import LinearScheduler




class DropPath_(nn.Module):
    def __init__(self, drop_prob):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training:
            if self.drop_prob >= 0. and self.drop_prob<=1.:
                keep_prob = 1. - self.drop_prob
                mask = torch.zeros(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device)
                x=x.div(keep_prob).mul(mask)
            else:
                raise ValueError('drop_prob must be positive number between 0~1')

        return x


class ScheduleDropPath(nn.Module):
    def __init__(self,start_dropout_rate, stop_dropout_rate, steps):
        super(ScheduleDropPath,self).__init__()
        self.droppath=DropPath_(start_dropout_rate)
        self.schduler=LinearScheduler(self.droppath,start_dropout_rate,stop_dropout_rate,steps)
        self.start_dropout_rate=start_dropout_rate
        self.stop_dropout_rate=stop_dropout_rate

    def forward(self,x):
        return self.schduler(x)
