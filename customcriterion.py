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
        
#       assert xpred.shape == xgt.shape 
#       assert ypred.shape == ygt.shape
        ind = [x for x in range(xpred.shape[1])]
        #print(len(ind),ind)
        c = True    
        for i in ind:
            if c:
                loss = torch.mean((xpred[:,i].unsqueeze(1)-xgt[:,(int)(i*10):(int)((i+1)*10)])**4+(ypred[:,i].unsqueeze(1)-ygt[:,(int)(i*10):(int)((i+1)*10)])**4)
                c = False
            loss += torch.mean((xpred[:,i].unsqueeze(1)-xgt[:,(int)(i*10):(int)((i+1)*10)])**4+(ypred[:,i].unsqueeze(1)-ygt[:,(int)(i*10):(int)((i+1)*10)])**4)
        loss = loss/len(ind)
        return loss