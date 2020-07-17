import numpy as np
import pylab as plt

def circle_matrix(radius=10, side=256, w=60):
    xx, yy = np.mgrid[:side, :side]
    #print(xx)
    #print(type(xx))
    x = np.random.randint(side)
    y = np.random.randint(side)
    circle = (xx - x) ** 2 + (yy - y) ** 2
    R = radius**2/4
    w = 600/6400*R
    donut = np.logical_and(circle < (R + w), circle > (R - w))
    values = {'donut': donut, 'x': x, 'y': y}
    print(values)
    return values


def plot_all(R=5, side=32, w=10):
    result = circle_matrix(radius=2*R, side=side, w=w)
    plt.imshow(result['donut'], cmap=plt.cm.gray_r)
    a_circle = plt.Circle((result['y'], result['x']), R, edgecolor='r', facecolor=None, fill=False)
    plt.gca().add_artist(a_circle)


for i in range(100):
    plt.subplot(10,10,i+1)
    plot_all(R=np.random.randint(30))
    plt.axis('off')
plt.savefig('finalplot.jpg')
