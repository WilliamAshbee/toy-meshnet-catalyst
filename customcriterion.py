import torch
from torch.nn.modules.loss import _Loss


class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        batchsize = input.shape[0]
        assert input.shape == (batchsize,6)
        assert target.shape == (batchsize,6)
        xgt = target[:,0]
        ygt = target[:,1]
        
        xpred = input[:,0]
        ypred = input[:,1]
        
        loss_xy = (xgt-xpred)**2+(ygt-ypred)**2
        loss_xy = torch.sum(loss_xy)/(float)(batchsize)

        rpred = input[:,-4:]
        rgt = target[:,-4:]
        assert rpred.shape == (batchsize,4)
        assert rgt.shape == (batchsize,4)
        
        xrfactors = torch.zeros_like(rpred)
        yrfactors = torch.zeros_like(rpred)
        xrfactors[:,0] = 0.0
        xrfactors[:,1] = -1.0
        xrfactors[:,2] = 0.0
        xrfactors[:,3] = 1.0
        
        yrfactors[:,0] = 1.0
        yrfactors[:,1] = 0.0
        yrfactors[:,2] = -1.0
        yrfactors[:,3] = 0.0
        
        xpred = xpred.unsqueeze(1)
        ypred = ypred.unsqueeze(1)
        
        xpred = xpred+xrfactors*rpred
        ypred = ypred+yrfactors*rpred

        xgt = xgt.unsqueeze(1)
        ygt = ygt.unsqueeze(1)
        assert rgt.shape == (batchsize,4)
        assert rpred.shape == (batchsize,4)
        assert xgt.shape == (batchsize,1)
        assert ygt.shape == (batchsize,1)
        xgt = xgt+xrfactors*rgt
        ygt = ygt+yrfactors*rgt

        loss_rpts = (xpred-xgt)**2+(ypred-ygt)**2
        assert loss_rpts.shape == (batchsize,4)
        loss_rpts = torch.sum(loss_rpts)/(batchsize*4.0)
        return loss_xy+loss_rpts
