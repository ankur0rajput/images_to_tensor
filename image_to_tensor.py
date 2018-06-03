# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 15:42:29 2018

@author: hp
"""
import os
import torch
from torchvision.transforms import transforms
from PIL import Image
#1  code for the coversion of image to tensor


t=0        
for i,filename in enumerate(os.listdir(r'F:\Projects\Datasets\Image\test_image\1.colors\silver color')):
#    print(i,filename)
    try:
        img=Image.open(filename)
        t=t+1
        img_tensor=transforms.ToTensor()(img)
        torch.save(img_tensor,r'F:\newest_dataset\silver_tensor\%d.silver'%(i))
    except:
        continue
print(t)


#2. to convert tensor to images

import os

from PIL import Image
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
for i,file in enumerate(os.listdir(r'/home/nibaran/newest_dataset_in_tensor/brown_tensor')):
    try:
        
#    print(file)
        im=torch.load(r'/home/nibaran/newest_dataset_in_tensor/brown_tensor'+'//'+file)
        save_image(im,r'/home/nibaran/new_dataset_in_images/1.colors/brown/%d.jpg'%(i))
    except:
        continue


#3.  code for check an image is openable or not

data=[] 
count=0
import cv2,glob
import os
images=glob.glob("*.jpg")
for image in images:
    try:
        img=cv2.imread(image,1)
#        img=Image.open(r'\Projects\Datasets\Image\train_image\1.colors\aegean color'+filename)
    #    re=cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
        cv2.imshow("checking",img)
        cv2.waitKey(10)
        cv2.destroyAllWindows()
         
    except:
#        os.remove(image)
        count=count+1
        data.append(image)
        
#        try:
#            os.remove(r'F:\Projects\Datasets\Image\train_image\1.colors\aegean color'+'/'+image)
#        except:
#            os.remove(r'\\?\UNC\F:\Projects\Datasets\Image\train_image\1.colors\aegean color'+'/'+image)
        continue
print(count)
print(data)    
