import torch
from torch.nn.modules.loss import _Loss


class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        batchsize = input.shape[0]
        assert input.shape == (batchsize,6)
        assert target.shape == (batchsize,3,2)
        #print('after assertion')
        xgt = target[:,:,0]
        ygt = target[:,:,1]
        assert xgt.shape == (batchsize,3)
        assert ygt.shape == (batchsize,3)
        #print('xgt',xgt.shape)

        xpred = input[:,:3]
        assert xpred.shape == (batchsize,3)
        ypred = input[:,-3:]
        assert ypred.shape == (batchsize,3)
        loss = (xpred-xgt)**2+(ypred-ygt)**2
        loss = torch.sum(loss)/(batchsize*3.0)
        return loss
