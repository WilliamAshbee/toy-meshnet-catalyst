import torch
import numpy as np
import pylab as plt
from skimage import filters

def circle_matrix():
    side = 32
    radiusMax=10
    w = 1
    radius = np.random.randint(1,radiusMax)
    sigmas = [None, 1, 2, 3, 4, 5]
    sigma=sigmas[np.random.randint(len(sigmas))]

    xx, yy = np.mgrid[:side, :side]
    x = np.random.randint(radius, side - radius)
    y = np.random.randint(radius, side - radius)
    #print('xyr',x,y,radius)
    assert x+radius <= side
    assert y+radius <= side
    assert x-radius >= 0
    assert y-radius >= 0
    
    #x = np.random.randint(side)
    #y = np.random.randint(side)
    circle = (xx - x) ** 2 + (yy - y) ** 2
    R2 = (radius-w)**2
    R1 = (radius+w)**2
    donut = np.logical_and(circle < R1, circle > R2)
    if sigma is not None:
        donut = filters.gaussian(donut, sigma=(sigma, sigma))
    
    return {'donut': donut, 'x': x, 'y': y, 'radius':radius}


def plot_all( sample = None, model = None):
    img = sample[0,:,:].squeeze().cpu().numpy()
    plt.imshow(img, cmap=plt.cm.gray_r)
    map = model(sample.unsqueeze(0).cuda())
    a_circle = plt.Circle((map[0,1], map[0,0]), map[0,2], edgecolor='r', facecolor=None, fill=False)
    plt.gca().add_artist(a_circle)


class DonutDataset(torch.utils.data.Dataset):
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
        map = circle_matrix()
        out = [map['x'],map['y'],map['radius']]
        result = map['donut']
        assert result.shape == (32,32)
        result = np.reshape(result,(1,32,32))
        assert result.shape == (1,32,32)
        
        result = torch.from_numpy(result)
        result = result.repeat(3, 1, 1).float()
        assert result.shape == (3,32,32)
        return result, torch.FloatTensor(out)
    
    @staticmethod
    def displayDonuts(dataset, model):
        for i in range(100):
            sample, _ = dataset[i]
            plt.subplot(10,10,i+1)
            plot_all(sample = sample,model=model)
            plt.axis('off')
        plt.savefig('finalplot.png')

#dataset = DonutDataset(length = 1024)
#DonutDataset.displayDonuts(dataset)