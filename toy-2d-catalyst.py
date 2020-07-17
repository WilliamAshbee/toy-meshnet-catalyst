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
    num_epochs=5,
    verbose=True,
    callbacks=[dl.BatchOverfitCallback(train=10, valid=10)]
)
print("finished running")
print('end of map')
from PIL import Image
import numpy as np

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
count = 0
for el, labels  in iter(loaders["train"]):
    #el = el*(el<torch.mean(el))
    print(torch.sum(el))
    #print('el0', el[0].shape)
    #print('el1', el[1].shape)
    #print('len el', len(el))
    #print(type(el))
    a = el
    
    predictions = model(a.cuda())
    for i in range(predictions.shape[0]):
        fig = plt.figure()
        ax = plt.gca()  # ax = subplot( 1,1,1 )
        count+=1
        ########
        xt = labels[i,0]
        yt = labels[i,1]
        rt = labels[i,2]
        e2 = Circle( xy=(xt, yt), radius= rt )
        ax.add_artist(e2)
        #print(ax.bbox)
        bb = ax.bbox
        bb._bbox = Bbox(np.array([[0.0, 0.0], [1.0, 1.0]], float))
        e2.set_clip_box(ax.bbox)
        e2.set_edgecolor( "black" )
        e2.set_facecolor( "none" )  # "none" not None
        e2.set_alpha( 1 )
        #plt.axis('off')
        base = '/home/users/washbee1/'
        
        ########
        x = predictions[i,0]
        y = predictions[i,1]
        r = predictions[i,2]
        name = "circleixyr_{}_{:.3f}_{:.3f}_{:.3f}_gt{:.3f}_{:.3f}_{:.3f}p".format(count,xt,yt,rt,x,y,r)
        e = Circle( xy=(x, y), radius= r )
        ax.add_artist(e)
        #print(ax.bbox)
        e.set_edgecolor( "red" )
        e.set_facecolor( "none" )  # "none" not None
        e.set_alpha( 1 )
        #plt.axis('off')
        base = '/home/users/washbee1/'
        plt.savefig("{}{}.jpg".format(base,name), bbox_inches='tight')
        plt.close(fig)
    

