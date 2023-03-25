import cv2
import os
import numpy as np

datapath = 'I:\\CodingProjects\\New_Rock_Paper_Scissors'
shape_to_label = {'rock':np.array([1.,0.,0.]),'paper':np.array([0.,1.,0.]),'scissors':np.array([0.,0.,1.])}

print(datapath)
for dr in os.listdir(datapath):
    #print(dr)
    if dr not in ['rock','paper','scissors']:
            continue
    for i,pic in enumerate(os.listdir(os.path.join(datapath,dr))):
        path = os.path.join(datapath,dr,pic)
        img = cv2.imread(path)
        img = cv2.flip(img,1)
        filename = 'new_'+dr+str(i+25)+'.jpg'
        filename = os.path.join(datapath,dr,filename)
        cv2.imwrite(filename,img)
