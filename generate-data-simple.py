from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import random
import csv
with open('labels-simple.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "i", "x","y","r"])
                
    for i in range(400):
        if i % 100 == 0:
            print(i)
        fig = plt.figure()

        x = random.uniform(.2, .8)
        y = random.uniform(.2, .8)
        r = random.uniform(.2, .8)
        name = "circleixyr_{}_{:.3f}_{:.3f}_{:.3f}".format(i,x,y,r)
        e = Circle( xy=(x, y), radius= r )
        ax = plt.gca()  # ax = subplot( 1,1,1 )
        ax.add_artist(e)
        #print(ax.bbox)
        bb = ax.bbox
        bb._bbox = Bbox(np.array([[0.0, 0.0], [1.0, 1.0]], float))
        e.set_clip_box(ax.bbox)
        e.set_edgecolor( "black" )
        e.set_facecolor( "none" )  # "none" not None
        e.set_alpha( 1 )
        plt.axis('off')
        base = '/data/mialab/users/washbee/circles-simple/'
        plt.savefig("{}{}.jpg".format(base,name), bbox_inches='tight')
        plt.close(fig)
        
        writer.writerow(["{}{}.jpg".format(base,name), i, x, y, r])