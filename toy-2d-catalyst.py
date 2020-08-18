import torch
from vgg import *
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from catalyst import dl
import numpy as np
from torch.utils import data
from torchvision import transforms
import argparse
#from CirclesLoad import CirclesLoad
from DonutDataset import DonutDataset
#from random_walk_dataset import RandomDataset
from loss import InpaintingLoss
from net import VGG16FeatureExtractor

from customcriterion import CustomCriterion
import torchvision.transforms.functional as F
import os
from resnet import resnet18, resnext101_32x8d
from inception import inception_v3
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print('data')
mini_batch = 200
#size = (32, 32)
parser = argparse.ArgumentParser()

parser.add_argument('--root', type=str, default='/data/mialab/users/washbee/circles-simple/')
args = parser.parse_args()

dataset_train = DonutDataset(40000)
dataset_val = DonutDataset(256)

loader_train = data.DataLoader(
    dataset_train, batch_size=mini_batch,
    sampler=RandomSampler(data_source = dataset_train),
    num_workers=4)

loader_val = data.DataLoader(
    dataset_val, batch_size=mini_batch,
    sampler=RandomSampler(data_source = dataset_val),
    num_workers=4)

loaders = {"train": loader_train, "valid": loader_val}

print('model')

# model, criterion, optimizer, scheduler
#model = vgg13().cuda()
#model = resnet18(pretrained=False, progress=True).cuda()
#model = inception_v3().cuda()
model = resnext101_32x8d().cuda()
#criterion = CustomCriterion().cuda()
criterion = InpaintingLoss(VGG16FeatureExtractor()).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay = 0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones =  [3,6,9,12,18,24,30,40,50,60,70,80,90], gamma = .5)

print('training')

# model training
runner = dl.SupervisedRunner()
logdir = './logdir'
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=70,
    verbose=True,
    callbacks=[dl.BatchOverfitCallback(train=10, valid=10)]
)

DonutDataset.displayCanvas(dataset_val,model)
