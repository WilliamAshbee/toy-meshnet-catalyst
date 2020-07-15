import torch
from vgg import *
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl
import numpy as np
from torch.utils import data
from torchvision import transforms
import argparse
from CirclesLoad import CirclesLoad
from customcriterion import CustomCriterion

print('data')
mini_batch = 512
size = (32, 32)
parser = argparse.ArgumentParser()

class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

img_tf = transforms.Compose(
    [
        transforms.Resize(size=size),
        transforms.ToTensor()
    ]
)

parser.add_argument('--root', type=str, default='/data/mialab/users/washbee/circles/')
args = parser.parse_args()

dataset_train = CirclesLoad(args.root,  img_tf, 'train',1000)
dataset_val = CirclesLoad(args.root,  img_tf, 'val',1000)

loader_train = data.DataLoader(
    dataset_train, batch_size=mini_batch,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=4)

loader_val = data.DataLoader(
    dataset_train, batch_size=mini_batch,
    sampler=InfiniteSampler(len(dataset_val)),
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
    callbacks=[dl.BatchOverfitCallback(train=10, valid=0.5)]
)

runner.valid_metrics = {"loss": criterion}
runnerpredictions = np.vstack(list(map(
    lambda x: x["logits"].cpu().numpy(), 
    runner.predict_loader(loader=loaders["valid"], resume=f"{logdir}/checkpoints/best.pth")
)))

print(runnerpredictions.shape)

secondRunnerPredictions = list(
    map(
        lambda x:x, 
        loaders["valid"]
        )
    )

for el in secondRunnerPredictions:
    print('el0', el[0].shape)
    print('el1', el[1].shape)
    print('len el', len(el))
    print('model',model(el[0]).shape)
    break

print(len(secondRunnerPredictions))
