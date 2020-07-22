import torch
from torch.nn.modules.loss import _Loss


class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        batchsize = input.shape[0]
        assert input.shape == (batchsize,6)
        assert target.shape == (batchsize,3)
        #print('after assertion')
        xgt = target[:,0].unsqueeze(1)
        ygt = target[:,1].unsqueeze(1)
        rgt = target[:,2].unsqueeze(1)
        #print('xgt',xgt.shape)

        xpred = input[:,:3]
        assert xpred.shape == (batchsize,3)
        ypred = input[:,-3:]
        assert ypred.shape == (batchsize,3)
        xbroadcastsub = xpred-xgt
        assert xbroadcastsub.shape == (batchsize,3)
        ybroadcastsub = ypred-ygt
        assert ybroadcastsub.shape == (batchsize,3)
        #print('after prediction assignment')
        loss = torch.abs(torch.sqrt((xbroadcastsub)**2+(ybroadcastsub)**2)-rgt)
        loss = torch.sum(loss)/(batchsize*3.0)
        print(loss)
        #x0 to x1
        innerDistance = torch.sum(torch.sqrt((xpred[:,0]-xpred[:,1])**2+(ypred[:,0]-ypred[:,1])**2))
        #x0 to x2
        innerDistance += torch.sum(torch.sqrt((xpred[:,0]-xpred[:,2])**2+(ypred[:,0]-ypred[:,2])**2))
        #x1 to x2
        innerDistance += torch.sum(torch.sqrt((xpred[:,1]-xpred[:,2])**2+(ypred[:,1]-ypred[:,2])**2))
        innerDistance = innerDistance/(batchsize*3.0)
        print(innerDistance)
        return loss-.2*innerDistance
