import torch
from torch.nn.modules.loss import _Loss


class CustomCriterion(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomCriterion, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        loss = torch.mean(torch.abs(input-target))
        return loss
