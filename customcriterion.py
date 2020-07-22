import torch
from torch.nn.modules.loss import _Loss
import math

class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        batchsize = input.shape[0]
        numpoints = input.shape[1]-2
        assert input.shape == (batchsize,numpoints+2)
        assert target.shape == (batchsize,numpoints+2)
        xgt = target[:,0]
        ygt = target[:,1]
        
        xpred = input[:,0]
        ypred = input[:,1]
        
        loss_xy = (xgt-xpred)**2+(ygt-ypred)**2
        loss_xy = torch.sum(loss_xy)/(float)(batchsize)

        rpred = input[:,-numpoints:]
        rgt = target[:,-numpoints:]
        assert rpred.shape == (batchsize,numpoints)
        assert rgt.shape == (batchsize,numpoints)
        
        xrfactors = torch.zeros_like(rpred)
        yrfactors = torch.zeros_like(rpred)
        theta = torch.FloatTensor(range(numpoints))
        theta*=1.0/numpoints
        theta*=math.pi*2.0
        
        xrfactors[:,:] = torch.cos(theta)
        yrfactors[:,:] = torch.sin(theta)
        
        xpred = xpred.unsqueeze(1)
        ypred = ypred.unsqueeze(1)
        
        xpred = xpred+xrfactors*rpred
        ypred = ypred+yrfactors*rpred

        xgt = xgt.unsqueeze(1)
        ygt = ygt.unsqueeze(1)
        assert rgt.shape == (batchsize,numpoints)
        assert rpred.shape == (batchsize,numpoints)
        assert xgt.shape == (batchsize,1)
        assert ygt.shape == (batchsize,1)
        xgt = xgt+xrfactors*rgt
        ygt = ygt+yrfactors*rgt

        loss_rpts = (xpred-xgt)**2+(ypred-ygt)**2
        assert loss_rpts.shape == (batchsize,numpoints)
        loss_rpts = torch.sum(loss_rpts)/(batchsize*numpoints)
        return loss_xy+loss_rpts
