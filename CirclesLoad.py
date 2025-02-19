import random
import torch
from PIL import Image
from glob import glob
import os
import numpy as np
import csv
class CirclesLoad(torch.utils.data.Dataset):
    def __init__(self, img_root, img_transform,
                 split='train', numExamples = None):
        super(CirclesLoad, self).__init__()
        self.numExamples = numExamples
        self.myvals = None
        with open('/home/users/washbee1/projects/toy-meshnet-catalyst/labels-simple.csv',mode = 'r') as infile:
            reader = csv.reader(infile)
            myvals = dict()
            skip = True
            for rows in reader:
                if skip:
                    skip = False
                    continue
                
                k = rows[0]
                if type(k) != type('str'):
                    assert False
                x = (float)(rows[2])
                y = (float)(rows[3])
                r = (float)(rows[4])
                myvals[k] = torch.FloatTensor([x,y,r])
            
            self.myvals = myvals

        self.img_transform = img_transform
        # use about 8M images in the challenge dataset
        self.paths = glob('{:s}*.jpg'.format(img_root))
        if len(self.paths )== 0:
            assert False
        if self.numExamples != None:
            self.paths = self.paths[:self.numExamples-1]
            print("length", len(self.paths))
        if split == 'train' :
            self.paths = self.paths[:(int)(len(self.paths)*.8)]
        else:
            self.paths = self.paths[-1*(int)(len(self.paths)*.2):]
    def getIndex(self,path):
        ind = -1
        for p in self.paths:
            ind+=1
            if p == path:
                return ind
        return -1
    

    def __getitem__(self, index):
        #print('index',index)
        path = self.paths[index]
        gt_img = Image.open(path)
        gt_img = self.img_transform(gt_img)
        #gt_img.save('test.jpg')
        #gt_img = gt_img.permute(2,1,0)
        #print(gt_img.shape)
        assert gt_img.shape == (3,32,32)
        return gt_img, self.myvals[path]

    def __len__(self):
        return len(self.paths)
