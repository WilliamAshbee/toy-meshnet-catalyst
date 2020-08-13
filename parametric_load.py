import math
import torch
import pyplot as plt

class ParametricDataset(torch.utils.data.Dataset):
    def __init__(self, length = 10):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        #TODO