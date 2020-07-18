import torch
import numpy as np
import pylab as plt

def circle_matrix(radius=10, side=256, w=60):
    xx, yy = torch.meshgrid(torch.arange(side),torch.arange(side))
    xx = xx
    yy = yy
    x = np.random.randint(side)
    y = np.random.randint(side)
    circle = (xx - x) ** 2 + (yy - y) ** 2
    R = radius**2/4
    w = 600/6400*R
    donut = torch.logical_and(circle < (R + w), circle > (R - w))
    values = {'donut': donut, 'x': x, 'y': y, 'radius':radius}
    return values

def plot_all(R=5, side=32, w=10,sample = None, model = None):
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
        R=5
        side=32
        w=10
        map = circle_matrix(radius=2*R, side=side, w=w)
        out = [map['x'],map['y'],map['radius']]
        result = map['donut'].repeat(3, 1, 1).float()
        assert result.shape == (3,32,32)
        return result, torch.FloatTensor(out)
    
    @staticmethod
    def displayDonuts(dataset, model):
        for i in range(100):
            sample, _ = dataset[i]
            plt.subplot(10,10,i+1)
            plot_all(sample = sample,model=model)
            plt.axis('off')
        plt.savefig('finalplot.jpg')

#dataset = DonutDataset(length = 1024)
#DonutDataset.displayDonuts(dataset)