import torch
from vgg import *
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from catalyst import dl
import numpy as np
from torch.utils import data
from torchvision import transforms
import argparse
from CirclesLoad import CirclesLoad
from DonutDataset import DonutDataset
from customcriterion import CustomCriterion
import torchvision.transforms.functional as F
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print('data')
mini_batch = 100
size = (32, 32)
parser = argparse.ArgumentParser()

img_tf = transforms.Compose(
    [
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0*(x<torch.mean(x)))
    ]
)
topil_tf = transforms.Compose(
    [
        transforms.ToPILImage()
    ]
)

parser.add_argument('--root', type=str, default='/data/mialab/users/washbee/circles-simple/')
args = parser.parse_args()

#dataset_train = CirclesLoad(args.root,  img_tf, 'train',None)
#dataset_val = CirclesLoad(args.root,  img_tf, 'val',None)
dataset_train = DonutDataset(256*32)
dataset_val = DonutDataset(256)


loader_train = data.DataLoader(
    dataset_train, batch_size=mini_batch,
    sampler=RandomSampler(data_source = dataset_train),
    num_workers=4)

loader_val = data.DataLoader(
    dataset_train, batch_size=mini_batch,
    sampler=RandomSampler(data_source = dataset_val),
    num_workers=4)

loaders = {"train": loader_train, "valid": loader_val}

print('model')

# model, criterion, optimizer, scheduler
model = vgg13().cuda()
criterion = CustomCriterion().cuda()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

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
    num_epochs=15,
    verbose=True,
    callbacks=[dl.BatchOverfitCallback(train=10, valid=10)]
)


DonutDataset.displayDonuts(dataset_val,model)