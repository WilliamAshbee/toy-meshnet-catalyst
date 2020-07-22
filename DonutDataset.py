import torch
import numpy as np
import pylab as plt
from skimage import filters

def circle_matrix():
    side = 32
    radiusMax=10
    w = 1
    radius = np.random.randint(1,radiusMax)
    sigmas = [None, 1]
    sigma=sigmas[np.random.randint(len(sigmas))]
    
    xx, yy = np.mgrid[:side, :side]
    x = np.random.randint(radius+5, side - radius-5)
    y = np.random.randint(radius+5, side - radius-5)
    a = torch.zeros((6,))
    a[0] = x
    a[1] = y
    a[2] = radius
    a[3] = radius
    a[4] = radius
    a[5] = radius
    
    #print('xyr',x,y,radius)
    
    circle = (xx - x) ** 2 + (yy - y) ** 2
    R2 = (radius-w)**2
    R1 = (radius+w)**2
    donut = np.logical_and(circle < R1, circle > R2)
    if sigma is not None:
        donut = filters.gaussian(donut, sigma=(sigma, sigma))
    
    return {'donut': donut, 'x': x, 'y': y, 'radius':radius, 'points':a}


def plot_all( sample = None, model = None, labels = None, circle = False):
    img = sample[0,:,:].squeeze().cpu().numpy()
    plt.imshow(img, cmap=plt.cm.gray_r)
    with torch.no_grad():
        pred = model(sample.unsqueeze(0).cuda())
        assert pred.shape == (1,6)
        xpred = pred[0,0].unsqueeze(0)
        ypred = pred[0,1].unsqueeze(0)
        assert xpred.shape == (1,)
        assert ypred.shape == (1,)
        rpred = pred[0,-4:]
        assert rpred.shape == (4,)
        xrfactors = torch.zeros_like(rpred)
        yrfactors = torch.zeros_like(rpred)
        xrfactors[0] = 0.0
        xrfactors[1] = -1.0
        xrfactors[2] = 0.0
        xrfactors[3] = 1.0
        yrfactors[0] = 1.0
        yrfactors[1] = 0.0
        yrfactors[2] = -1.0
        yrfactors[3] = 0.0
        #print(xpred,xrfactors.shape,rpred.shape)
        #assert False
        xpreds = xpred+xrfactors*rpred
        assert xpreds.shape == (4,)
        #assert False
        ypreds = ypred+yrfactors*rpred
        assert ypreds.shape == (4,)

        #print (ypreds.shape)
        #assert False
        xpreds = xpreds.cpu().numpy()
        assert xpreds.shape == (4,)
        ypreds = ypreds.cpu().numpy()
        assert ypreds.shape == (4,)
        
        X = xpred.cpu().numpy()
        Y = ypred.cpu().numpy()
        #print(X.shape)
        X = np.concatenate((X,xpreds),axis = 0)
        Y = np.concatenate((Y,ypreds),axis = 0)
        print(X)
        assert X.shape == (5,)
        assert Y.shape == (5,)
        # Plotting point using sactter method
        ascatter = plt.scatter(Y,X,s = [.6,.6,.6,.6,.6],c = ['blue','red','red','red','red'])
        plt.gca().add_artist(ascatter)
    """if circle:
        if model != None:
            map = model(sample.unsqueeze(0).cuda())
            x = map[0,0]
            y = map[0,1]
            r = map[0,2]
        elif label != None:
            x = labels[0]
            y = labels[1]
            r = labels[2]
        else:
            assert False,"Need eith model or gt labels"
        a_circle = plt.Circle((y, x), r, edgecolor='r', facecolor=None, fill=False)
        plt.gca().add_artist(a_circle)
    else:
        if model != None:
            with torch.no_grad():
                pred = model(sample.unsqueeze(0).cuda())
                xpred = pred[0,:3]
                ypred = pred[0,-3:]
                X = xpred.cpu().numpy()
                Y = ypred.cpu().numpy()
                assert X.shape == (3,)
                assert Y.shape == (3,)
                # Plotting point using sactter method
                ascatter = plt.scatter(Y,X,s = [.2,.2,.2])
                #a_circle = plt.Circle((y, x), r, edgecolor='r', facecolor=None, fill=False)
                plt.gca().add_artist(ascatter)
        else:
            
            X = labels[:,0].cpu().numpy()
            Y = labels[:,1].cpu().numpy()
            print(X,Y)
            assert X.shape[0] == 3
            assert Y.shape[0] == 3
            # Plotting point using sactter method
            ascatter = plt.scatter(Y,X,s = [.2,.2,.2])
            #a_circle = plt.Circle((y, x), r, edgecolor='r', facecolor=None, fill=False)
            plt.gca().add_artist(ascatter)
    """


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
        out = map['points']
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
            sample, labels = dataset[i]
            plt.subplot(10,10,i+1)
            plot_all(sample = sample,model=model, labels = labels)
            plt.axis('off')
        plt.savefig('finalplot.png')

#dataset = DonutDataset(length = 1024)
#DonutDataset.displayDonuts(dataset, model = None)
