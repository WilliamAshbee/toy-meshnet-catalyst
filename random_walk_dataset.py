import torch
import numpy as np
import pylab as plt
from skimage import filters
import math

global numpoints
numpoints = 1000
modn = 10
side = 16

def random_matrix(length = 10):
    radiusMax = side /3
    w = 1
    sigmas = [None, 1]
    
    canvas = torch.zeros((length,side, side))
    x0 = np.random.uniform(1+radiusMax, side - radiusMax-1,length)
    y0 = np.random.uniform(1+radiusMax, side - radiusMax-1,length)
    r0 = np.random.uniform(2, radiusMax, length)

    radii = np.zeros((length,numpoints))    
    radii[:, 0] = r0
    for i in range(1,numpoints):
        #print(radii[:, i-1].shape)
        radii[:, i] = radii[:, i-1] + np.random.uniform(-1.0,1.0,(length))
        radii[radii[:,i-1]<= 2, i] = radii[radii[:,i-1]<= 2.0, i-1] + np.random.uniform(0.0,1.0,np.sum(radii[:,i-1]<= 2.0))
        radii[radii[:,i-1]>= radiusMax-1, i] = radii[radii[:,i-1]>= radiusMax-1, i-1] + np.random.uniform(-1.0,0.0,np.sum(radii[:,i-1] >= radiusMax-1))
    #print(radii)
    ind = [x for x in range(numpoints)]
    theta = torch.FloatTensor(ind)
    theta *= math.pi*2.0/(float)(numpoints)
    
    x0 = torch.from_numpy(x0).unsqueeze(1)
    y0 = torch.from_numpy(y0).unsqueeze(1)
    radii = torch.from_numpy(radii)
    xrfactors = torch.cos(theta).unsqueeze(0)
    yrfactors = torch.sin(theta).unsqueeze(0)
    
    print(x0.shape,y0.shape,radii.shape,xrfactors.shape,yrfactors.shape)

    x = (x0+(xrfactors*radii))
    y = (y0+(yrfactors*radii))
    assert x.shape == (length,numpoints)
    assert y.shape == (length,numpoints)
    assert torch.sum(x[x>(side-1)])==0 
    assert torch.sum(x[x<0])==0 
    assert torch.sum(y[y>(side-1)])==0 
    assert torch.sum(y[y<0])==0 
    
    points = torch.zeros(length,numpoints,2)
    for l in range(length):
        
        canvas[l,x[l,:].type(torch.LongTensor),y[l,:].type(torch.LongTensor)]=1.0
        points[l,:,0] = x[l,:]
        points[l,:,1] = y[l,:]
    
    return {
        'canvas': canvas, 
        'points':points.type(torch.FloatTensor)}


def plot_all( sample = None, model = None, labels = None):
    img = sample[0,:,:].squeeze().cpu().numpy()
    plt.imshow(img, cmap=plt.cm.gray_r)
    if model != None:
        with torch.no_grad():
            global numpoints

            pred = model(sample.unsqueeze(0).cuda())
            X = pred[0,:numpoints]
            Y = pred[0,-numpoints:]
            #print (X.shape,Y.shape)
            s = [.1 for x in range(numpoints)]
            assert len(s) == numpoints
            c = ['red' for x in range(numpoints)]
            assert len(c) == numpoints
            ascatter = plt.scatter(Y.cpu().numpy(),X.cpu().numpy(),s = s,c = c)
            plt.gca().add_artist(ascatter)
    else:
        #print(labels.shape)

        X = labels[:numpoints,0]
        Y = labels[:numpoints,1]
        s = [.001 for x in range(numpoints)]
        c = ['red' for x in range(numpoints)]
        ascatter = plt.scatter(Y.cpu().numpy(),X.cpu().numpy(),s = s,c = c)
        plt.gca().add_artist(ascatter)

class RandomDataset(torch.utils.data.Dataset):
    """Donut dataset."""
    def __init__(self, length = 10):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.length = length
        self.values = random_matrix(length)
        assert self.values['canvas'].shape[0] == self.length
        assert self.values['points'].shape[0] == self.length

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        canvas = self.values["canvas"]
        
        canvas = canvas[idx,:,:]
        assert canvas.shape == (side,side)
        canvas = torch.reshape(canvas,(1,side,side))
        assert canvas.shape == (1,side,side)
        
        #canvas = torch.from_numpy(canvas)
        canvas = canvas.repeat(3, 1, 1).float()
        assert canvas.shape == (3,side,side)

        points = self.values["points"]
        points = points[idx,:,:]
        #points = torch.from_numpy(points)
        assert points.shape == (numpoints,2)
        
        return canvas, points
    
    @staticmethod
    def displayCanvas(dataset, model):
        for i in range(100):
            sample, labels = dataset[i]
            plt.subplot(10,10,i+1)
            plot_all(sample = sample,model=model, labels = labels)
            plt.axis('off')
        plt.savefig('finalplot.png',dpi=600)

dataset = RandomDataset(length = 100)
RandomDataset.displayCanvas(dataset, model = None)
