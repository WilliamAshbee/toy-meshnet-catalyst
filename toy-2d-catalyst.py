import torch
from vgg import *
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from catalyst import dl
import numpy as np
from torch.utils import data
from torchvision import transforms
import argparse
from CirclesLoad import CirclesLoad
from customcriterion import CustomCriterion
import torchvision.transforms.functional as F


print('data')
mini_batch = 100
size = (32, 32)
parser = argparse.ArgumentParser()


img_tf = transforms.Compose(
    [
        transforms.Resize(size=size),
        transforms.ToTensor()
    ]
)
topil_tf = transforms.Compose(
    [
        transforms.ToPILImage()
    ]
)

parser.add_argument('--root', type=str, default='/data/mialab/users/washbee/circles-simple/')
args = parser.parse_args()

dataset_train = CirclesLoad(args.root,  img_tf, 'train',None)
dataset_val = CirclesLoad(args.root,  img_tf, 'val',None)

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
    num_epochs=1,
    verbose=True,
    callbacks=[dl.BatchOverfitCallback(train=10, valid=10)]
)

print("finished running")

stats = True
secondRunnerPredictions = list(
    map(
        lambda x:x, 
        iter(loaders["valid"])
        )
    )
print('end of map')
from PIL import Image
import numpy as np

for el in secondRunnerPredictions:
    print('el0', el[0].shape)
    print('el1', el[1].shape)
    print('len el', len(el))
    print('model',model(el[0].cuda()).shape)
    ind = 1
    print(type(el))
    a = el[0][ind,:,:,:]
    #a = a * 255.0
    #a = a.reshape((32,32,3))
    #a = np.transpose(a,(2,1,0))
    #a = np.transpose(a, (1,2,0))
    
    print(a)
    print(a.shape)
    a = a.to(torch.device("cpu"))
    img = topil_tf(a)
    print (type(img))
    img.save('pil.jpg')
    print('label',el[1][ind])
    break

print(len(secondRunnerPredictions))
