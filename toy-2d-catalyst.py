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


print('data')
mini_batch = 100
size = (32, 32)
parser = argparse.ArgumentParser()

finiteS = 1000
stats = False

img_tf = transforms.Compose(
    [
        transforms.Resize(size=size),
        transforms.ToTensor()
    ]
)

parser.add_argument('--root', type=str, default='/data/mialab/users/washbee/circles-v2/')
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
    num_epochs=10,
    verbose=True,
    callbacks=[dl.BatchOverfitCallback(train=10, valid=10)]
)
print("finished running")
"""
runner.valid_metrics = {"loss": criterion}
runnerpredictions = np.vstack(list(map(
    lambda x: x["logits"].cpu().numpy(), 
    runner.predict_loader(loader=loaders["valid"], resume=f"{logdir}/checkpoints/best.pth")
)))

print(runnerpredictions.shape)
"""
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
    a = el[0][0,:,:,:]
    a = a.reshape((32,32,3))
    a = a.numpy()
    print(a)
    print(a.shape)
    img = Image.fromarray(a, 'RGB')
    img.save('my.png')

    break

print(len(secondRunnerPredictions))
