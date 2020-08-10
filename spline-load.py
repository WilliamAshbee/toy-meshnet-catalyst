import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt


def bspline(cv, n=100, degree=3):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
    """
    print ("cv",cv)
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Prevent degree from exceeding count-1, otherwise splev will crash
    degree = np.clip(degree,1,count-1)
    print("degree", degree)
    # Calculate knot vector
    kv = np.array([0]*degree + [x for x in range(count-degree+1)] + [count-degree]*degree,dtype='int')
    print("kv",kv)
    # Calculate query range
    u = np.linspace(0,(count-degree),n)
    result = np.array(si.splev(u, (kv,cv.T,degree))).T
    print(result)
    # Calculate result
    return result

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

cv = np.array([[ 50.,  25.],
   [ 59.,  12.],
   [ 50.,  10.],
   [ 57.,   2.],
   [ 40.,   4.],
   [ 40.,   14.]])

plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')#plots control points

for d in range(1,5):
    p = bspline(cv,n=100,degree=d)
    x,y = p.T
    plt.plot(x,y,'k-',label='Degree %s'%d,color=colors[d%len(colors)])#plots fit points

plt.minorticks_on()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
#plt.xlim(35, 70)
#plt.ylim(0, 30)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('finalplot.png',dpi=300)
