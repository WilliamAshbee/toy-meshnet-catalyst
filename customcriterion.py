import torch
from torch.nn.modules.loss import _Loss
import math

class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)
        self.i = 0
    def forward(self, input, target):
        xpred = input[:,:100]
        ypred = input[:,-100:]
        
        xgt = target[:,:,0]
        ygt = target[:,:,1]
        
        assert xpred.shape == xgt.shape 
        assert ypred.shape == ygt.shape
        
        loss = torch.mean((xpred-xgt)**2+(ypred-ygt)**2)
        return loss
