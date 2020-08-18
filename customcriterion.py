import torch
from torch.nn.modules.loss import _Loss
import math

import torch
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)
        self.i = 0
        self.resnet=False
    def forward(self, input, target):
        #print ('xpred',type(input[0]))
        if self.resnet:
            try:
                xpred = input[0][:,:1000]
                ypred = input[0][:,-1000:]
            except:
                print(input.shape)
                xpred = input[:,:1000]
                ypred = input[:,-1000:]
        else:
            xpred = input[:,:1000]
            ypred = input[:,-1000:]
            
        
        xgt = target[:,:,0]
        ygt = target[:,:,1]
        
        assert xpred.shape == xgt.shape 
        assert ypred.shape == ygt.shape
        ind = [x for x in range(xpred.shape[1])]
        #print(len(ind),ind)
        loss = torch.mean((xpred-xgt)**4+(ypred-ygt)**4)
        return loss
        """
        c = True    
        for i in ind:
            if c:
                loss = torch.mean((xpred[:,i].unsqueeze(1)-xgt[:,(int)(i*10):(int)((i+1)*10)])**4+(ypred[:,i].unsqueeze(1)-ygt[:,(int)(i*10):(int)((i+1)*10)])**4)
                c = False
            loss += torch.mean((xpred[:,i].unsqueeze(1)-xgt[:,(int)(i*10):(int)((i+1)*10)])**4+(ypred[:,i].unsqueeze(1)-ygt[:,(int)(i*10):(int)((i+1)*10)])**4)
        loss = loss/len(ind)
        return loss
        """