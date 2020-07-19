import numpy as np
import pylab as plt
from skimage import filters


def circle_matrix(radius=10, side=256, w=5, sigma=None):
    xx, yy = np.mgrid[:side, :side]
    x = np.random.randint(side)
    y = np.random.randint(side)
    circle = (xx - x) ** 2 + (yy - y) ** 2
    R2 = (radius-w)**2
    R1 = (radius+w)**2
    donut = np.logical_and(circle < R1, circle > R2)
    if sigma is not None:
        donut = filters.gaussian(donut, sigma=(sigma, sigma))
    return {'donut': donut, 'x': x, 'y': y}


def plot_all(R=5, side=64, w=10):
    sigmas = [None, 1, 2, 3, 4, 5]
    result = circle_matrix(radius=R, side=side, w=w,
                           sigma=sigmas[np.random.randint(len(sigmas))])
    plt.imshow(result['donut'], cmap=plt.cm.gray_r)
    a_circle = plt.Circle((result['y'], result['x']), R,
                          edgecolor='r', facecolor=None, fill=False)
    plt.gca().add_artist(a_circle)


side=64


for i in range(100):
    plt.subplot(10,10,i+1)
    plot_all(R=np.random.randint(side), side=side, w=2)
    plt.axis('off')


plt.savefig('finalplot.jpg')

