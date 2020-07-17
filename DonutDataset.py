import torch
import numpy as np
import pylab as plt



def circle_matrix(radius=10, side=256, w=60):
    xx, yy = torch.meshgrid(torch.arange(side),torch.arange(side))
    xx = xx.cuda()
    yy = yy.cuda()
    x = np.random.randint(side)
    y = np.random.randint(side)
    circle = (xx - x) ** 2 + (yy - y) ** 2
    R = radius**2/4
    w = 600/6400*R
    donut = torch.logical_and(circle < (R + w), circle > (R - w)).cuda()
    values = {'donut': donut, 'x': x, 'y': y}
    return values

def plot_all(R=5, side=32, w=10):
    result = circle_matrix(radius=2*R, side=side, w=w)
    plt.imshow(result['donut'].cpu().numpy(), cmap=plt.cm.gray_r)
    a_circle = plt.Circle((result['y'], result['x']), R, edgecolor='r', facecolor=None, fill=False)
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
        return map['donut']


dataset = DonutDataset(length = 1024)


for i in range(100):
    sample = dataset[i]
    plt.subplot(10,10,i+1)
    plot_all(R=np.random.randint(30))
    plt.axis('off')
plt.savefig('finalplot.jpg')