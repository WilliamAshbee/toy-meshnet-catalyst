import torch
from torch.nn.modules.loss import _Loss


class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        assert input.shape == target.shape
        loss = 2.0*torch.sum(torch.square(input[:,0]-target[:,0]))
        loss += 2.0*torch.sum(torch.square(input[:,1]-target[:,1]))
        loss += torch.sum(torch.square(input[:,2]-target[:,2]))
        return torch.sqrt(loss)
