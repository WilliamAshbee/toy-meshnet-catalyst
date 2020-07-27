import torch
from torch.nn.modules.loss import _Loss
import math

class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        numpoints = input.shape[1]-2
        ind = [x for x in range(numpoints)]
        assert numpoints == 100
        theta = torch.cuda.FloatTensor(ind)
        theta *= math.pi*2.0/(float)(numpoints)
        x0pred = input[:,0].unsqueeze(1)
        y0pred = input[:,1].unsqueeze(1)
        rpred = input[:,-numpoints:]
        #print(x0pred.shape,rpred.shape,theta.shape)
        
        xpred = x0pred+(torch.cos(theta)*rpred)

        ypred = y0pred+torch.sin(theta)*rpred
        
        x0gt = target[:,0].unsqueeze(1)
        y0gt = target[:,1].unsqueeze(1)
        rgt = target[:,-numpoints:]
        xgt = x0gt+torch.cos(theta)*rgt
        ygt = y0gt+torch.sin(theta)*rgt
        
        

        loss = torch.mean((xpred-xgt)**2+(ypred-ygt)**2)
        return loss
