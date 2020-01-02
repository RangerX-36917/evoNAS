import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: inefficient, see CrossEntropyLoss
class CrossEntropyLossMOD(nn.Module):
    def __init__(self, classes,smoothing=0.1):
        super(CrossEntropyLossMOD, self).__init__()
        self.classes=classes
        self.smoothing=smoothing


    def _onehot_and_smooth(self,indexes):
        batch_size=indexes.shape[0]
        onehot=torch.full((batch_size,self.classes),fill_value=self.smoothing/(self.classes-1) ,device=indexes.device)
        onehot.scatter_(-1,indexes.unsqueeze(1),1-self.smoothing)

        return onehot


    def forward(self, logits, target):
        pred=F.log_softmax(logits,dim=-1)
        smooth_target=self._onehot_and_smooth(target)
        return (-smooth_target * pred).sum(dim=-1).mean()

        