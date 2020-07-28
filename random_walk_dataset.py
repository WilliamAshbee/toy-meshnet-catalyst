import torch
import numpy as np
import pylab as plt
from skimage import filters
import math

numpoints = 100

def random_matrix():
    side = 32
    radiusMax = 14
    w = 1
    sigmas = [None, 1]

    #sigma=sigmas[np.random.randint(len(sigmas))]
    
    canvas = np.zeros((side, side))
    radius = np.random.randint(2, radiusMax-2)
    
    x0 = 16#np.random.randint(1+radiusMax, side - radiusMax)
    y0 = 16#np.random.randint(1+radiusMax, side - radiusMax)
    
    
    radii = np.zeros((numpoints))
    
    for i in range(numpoints):
        if radius <= 2:
            radius = radius + np.random.choice([0,1],1)
            radii[i] = radius
        elif radius >= radiusMax-1:
            radius = radius + np.random.choice([-1,0],1)
            radii[i] = radius
        else:
            radius = radius + np.random.choice([-1,0,1],1)
            radii[i] = radius
    
    ind = [x for x in range(numpoints)]
    theta = torch.FloatTensor(ind)
    theta *= math.pi*2.0/(float)(numpoints)
    
    xrfactors = torch.zeros(numpoints)
    yrfactors = torch.zeros(numpoints)
    
    xrfactors[:] = torch.cos(theta)
    yrfactors[:] = torch.sin(theta)
    x = (x0+xrfactors*radii).type(torch.LongTensor)
    y = (y0+yrfactors*radii).type(torch.LongTensor)
    assert torch.sum(x[x>31])==0 
    assert torch.sum(x[x<0])==0 
    assert torch.sum(y[y>31])==0 
    assert torch.sum(y[y<0])==0 
    canvas[x,y]=1.0
    #    for i in ind:
    #        x = (int)(x0+xrfactors[i]*radii[i])
    #        y = (int)(y0+yrfactors[i]*radii[i])
    #        assert x >= 0
    #        assert x <= 31
    #        assert y >= 0
    #        assert y <= 31
    #        canvas[x,y] = 1.0
        

    
    #if sigma is not None:
    #    donut = filters.gaussian(donut, sigma=(sigma, sigma))
    points = torch.zeros(x.shape[0],2)
    points[:,0] = x
    points[:,1] = y
    return {
        'canvas': canvas, 
        'x': x0, 
        'y': y0,
        'points':points.type(torch.FloatTensor)}


def plot_all( sample = None, model = None, labels = None):
    img = sample[0,:,:].squeeze().cpu().numpy()
    plt.imshow(img, cmap=plt.cm.gray_r)
    if model != None:
        with torch.no_grad():
            pred = model(sample.unsqueeze(0).cuda())
            x0 = pred[0,0].cpu()
            y0 = pred[0,1].cpu()
            rs = pred[0,-numpoints:].cpu()
            
            X = torch.zeros((numpoints+1,))
            Y = torch.zeros((numpoints+1,))
            
            X[0] = x0
            Y[0] = y0
            
            ind = [x for x in range(numpoints)]
            theta = torch.FloatTensor(ind)
            theta *= math.pi*2.0/(float)(numpoints)

            
            X[-numpoints:] = x0+torch.cos(theta)*rs[-numpoints:]
            Y[-numpoints:] = y0+torch.sin(theta)*rs[-numpoints:]
            s = [.6 for x in range(numpoints+1)]
            c = ['red' for x in range(numpoints+1)]
            c[0] = 'blue'
            ascatter = plt.scatter(Y.cpu().numpy(),X.cpu().numpy(),s = s,c = c)
            plt.gca().add_artist(ascatter)
    else:
        
        X = labels[:,0]
        Y = labels[:,1]
        s = [.1 for x in range(numpoints)]
        c = ['red' for x in range(numpoints)]
        ascatter = plt.scatter(Y.cpu().numpy(),X.cpu().numpy(),s = s,c = c)
        plt.gca().add_artist(ascatter)

class RandomDataset(torch.utils.data.Dataset):
    """Donut dataset."""
    def __init__(self, length = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.length = length


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        map = random_matrix()
        points = map['points']
        result = map['canvas']
        assert result.shape == (32,32)
        result = np.reshape(result,(1,32,32))
        assert result.shape == (1,32,32)
        
        result = torch.from_numpy(result)
        result = result.repeat(3, 1, 1).float()
        assert result.shape == (3,32,32)
        return result, points
    
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
