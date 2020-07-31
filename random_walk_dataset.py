import torch
import numpy as np
import pylab as plt
from skimage import filters
import math

numpoints = 100

def random_matrix(length = 10):
    side = 32
    radiusMax = 12
    w = 1
    sigmas = [None, 1]

    canvas = torch.zeros((length,side, side))
    x0 = np.array([np.random.randint(1+radiusMax, side - radiusMax-1) for x in range(length)])
    y0 = np.array([np.random.randint(1+radiusMax, side - radiusMax-1) for x in range(length)])
    r0 = np.array([np.random.randint(2, radiusMax) for x in range(length)])

    radii = np.zeros((length,numpoints))    
    radii[:, 0] = r0
    for i in range(1,numpoints):
        radii[:, i] = radii[:, i-1] + np.random.choice([-1,0,1],1)
        radii[radii[:,i-1]<= 2, i] = radii[radii[:,i-1]<= 2, i-1] + np.random.choice([0,1],1)
        radii[radii[:,i-1]>= radiusMax-1, i] = radii[radii[:,i-1]>= radiusMax-1, i-1] + np.random.choice([-1,0],1)
        
    ind = [x for x in range(numpoints)]
    theta = torch.FloatTensor(ind)
    theta *= math.pi*2.0/(float)(numpoints)
    
    x0 = torch.from_numpy(x0).unsqueeze(1)
    y0 = torch.from_numpy(y0).unsqueeze(1)
    radii = torch.from_numpy(radii)
    xrfactors = torch.cos(theta).unsqueeze(0)
    yrfactors = torch.sin(theta).unsqueeze(0)
    
    print(x0.shape,y0.shape,radii.shape,xrfactors.shape,yrfactors.shape)

    x = (x0+(xrfactors*radii)).type(torch.LongTensor)
    y = (y0+(yrfactors*radii)).type(torch.LongTensor)
    assert x.shape == (length,numpoints)
    assert y.shape == (length,numpoints)
    assert torch.sum(x[x>31])==0 
    assert torch.sum(x[x<0])==0 
    assert torch.sum(y[y>31])==0 
    assert torch.sum(y[y<0])==0 
    
    points = torch.zeros(length,numpoints,2)
    for l in range(length):
        canvas[l,x[l,:],y[l,:]]=1.0
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
            pred = model(sample.unsqueeze(0).cuda())
            numpoints = 100
            X = pred[0,:numpoints]
            Y = pred[0,-numpoints:]
            print (X.shape,Y.shape)
            s = [.1 for x in range(numpoints)]
            assert len(s) == numpoints
            c = ['red' for x in range(numpoints)]
            assert len(c) == numpoints
            ascatter = plt.scatter(Y.cpu().numpy(),X.cpu().numpy(),s = s,c = c)
            plt.gca().add_artist(ascatter)
    else:
        print(labels.shape)
        numpoints = 100
        X = labels[:numpoints,0]
        Y = labels[:numpoints,1]
        s = [.01 for x in range(numpoints)]
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
        print(type(canvas))
        
        canvas = canvas[idx,:,:]
        assert canvas.shape == (32,32)
        canvas = torch.reshape(canvas,(1,32,32))
        assert canvas.shape == (1,32,32)
        
        #canvas = torch.from_numpy(canvas)
        canvas = canvas.repeat(3, 1, 1).float()
        assert canvas.shape == (3,32,32)

        points = self.values["points"]
        points = points[idx,:,:]
        #points = torch.from_numpy(points)
        assert points.shape == (100,2)
        
        return canvas, points
    
    @staticmethod
    def displayCanvas(dataset, model):
        for i in range(100):
            sample, labels = dataset[i]
            plt.subplot(10,10,i+1)
            plot_all(sample = sample,model=model, labels = labels)
            plt.axis('off')
        plt.savefig('finalplot.png',dpi=600)

#dataset = RandomDataset(length = 1024)
#RandomDataset.displayCanvas(dataset, model = None)
