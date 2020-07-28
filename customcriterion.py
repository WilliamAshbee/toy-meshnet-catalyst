import torch
from torch.nn.modules.loss import _Loss
import math

class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)
        self.i = 0
    def forward(self, input, target):
        numpoints = input.shape[1]-2
        self.i+=1
        assert target.shape[1] == numpoints
        assert target.shape[2] == 2
        
        ind = [x for x in range(numpoints)]
        assert numpoints == 100
        theta = torch.cuda.FloatTensor(ind)
        theta *= math.pi*2.0/(float)(numpoints)
        x0pred = input[:,0].unsqueeze(1)
        y0pred = input[:,1].unsqueeze(1)
        
        if self.i%100 == 0:
            print("input is ",torch.sum(input))
            print(x0pred,y0pred)
        
        rpred = input[:,-numpoints:]
        #print(x0pred.shape,rpred.shape,theta.shape)
        
        xpred = x0pred+(torch.cos(theta)*rpred)

        ypred = y0pred+(torch.sin(theta)*rpred)
        
        xgt = target[:,:,0]
        ygt = target[:,:,1]
        

        assert xpred.shape == xgt.shape 
        assert ypred.shape == ygt.shape
        

        loss = torch.mean((xpred-xgt)**2+(ypred-ygt)**2)
        return loss
