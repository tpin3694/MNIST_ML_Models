# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:59:40 2017
studentid：32026312
@author: wangxinji
"""

# libraryloading images
import skimage.io as io
from skimage import data_dir
# library pocessing image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# using default dataset
str = data_dir + '/*.png'
coll = io.ImageCollection(str)
print(len(coll))
io.imshow(coll[10])
# loading picture from local
import skimage.io as io
from skimage import data_dir

str = '/home/tpin3694/Documents/university/MSc/dataMining/git/emotion_recognition/py/data/*.png'
coll = io.ImageCollection(str)
print(len(coll)) # Print number of images in directory
io.imshow(coll[0]) # Print image

# load the image as a gray picture
from skimage import data_dir, io, color


def convert_gray(f):
    rgb = io.imread(f)
    return color.rgb2gray(rgb)


coll = io.ImageCollection(str, load_func=convert_gray)
io.imshow(coll[0])

# turning image into np.array
img = np.array(coll[0])
img


def check(img):
    print(img.shape) # Dimensions
    print(img.dtype)
    print(img.size)
    type(img)
check(coll[0])

check(img)

# gray scale
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

str1 = 'C:/Users/Hasee/Desktop/datamining/data/pict/lena.jpg'
coll1 = io.ImageCollection(str1)

img = np.array(Image.open('C:/Users/Hasee/Desktop/datamining/data/pict/lena.jpg').convert('L'))
io.imshow(coll1[0])
img
check(img)
# turn it into a grayscale pict
rows, cols = img.shape
for i in range(rows):
    for j in range(cols):
        if (img[i, j] <= 120):
            img[i, j] = 0
        else:
            img[i, j] = 1
plt.figure('lena')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


