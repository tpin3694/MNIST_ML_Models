# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:59:40 2017
studentid：32026312
@author: wangxinji
"""

#libraryloading images
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from skimage import data_dir, io, color
import glob
from sklearn.decomposition import PCA

# Define directories
#dir_str='C:/Users/Hasee/Desktop/datamining/data/pict/*.png'
dir_str = "/home/tpin3694/Documents/university/MSc/dataMining/git/emotion_recognition/py/data/*.png"

# Function declaration
def convert_gray(f, img_num = 0):
    rgb = io.imread(f).astype(np.uint16)
    return color.rgb2gray(rgb)


def check(img_pic):
    print('dimmention' , img_pic.shape)
    print(img_pic.dtype)
    print(img_pic.size)
    type(img_pic)


def no_files(directory):
    n = len(glob.glob((directory)))
    return n


def pull_images(directory):
    collection = io.ImageCollection(directory, load_func=convert_gray)
    return collection

#using default dataset 
pic_str = data_dir + '/*.png'
coll = io.ImageCollection(pic_str)
io.imshow(coll[10])

# Locally load images
coll = pull_images(dir_str)

# Return how many images
no_imgs = no_files(dir_str)

print("There are " + str(no_imgs) + " files in the directory.")

# Iteratively build array of images
pic_array = []
for i in range(no_imgs):
    img = coll[i-1]
    img_array = img.flatten()
    pic_array.append(img_array)

# Turn list of arrays into matrix
out_array = np.array(pic_array)

# Run PCA

sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)  
pca = PCA(n_components=5000)       
pca_result = pca.fit(out_array)

print(pca_result.explained_variance_ratio_) 
plt.plot(np.cumsum(pca_result.explained_variance_ratio_))
#
# for i in range(n):
#     mean_out_array[i] -= mean_out_array