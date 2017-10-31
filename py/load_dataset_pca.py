# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:59:40 2017
studentid：32026312
@author: wangxinji
"""

#libraryloading images
import skimage.io as io
from skimage import data_dir
#library pocessing image 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sklearn

import os,os.path
#using default dataset 
str=data_dir + '/*.png'
coll = io.ImageCollection(str)
print(len(coll))
io.imshow(coll[10])
#loading picture from local
import skimage.io as io
from skimage import data_dir
str='C:/Users/Hasee/Desktop/datamining/data/pict/*.png'
coll = io.ImageCollection(str)
print(len(coll))
io.imshow(coll[1])

#load the image as a gray picture
from skimage import data_dir,io,color
def convert_gray(f):
    rgb = io.imread(f)
    return color.rgb2gray(rgb)

coll = io.ImageCollection(str,load_func=convert_gray)
io.imshow(coll[0])
x = coll[0].flatten()
type(x)
     
 #turning image into np.array
 
img = np.array(coll[1])
img
def check(img):
    print('dimmention' , img.shape)
    print(img.dtype)
    print(img.size)
    type(img)

check(img)


#print the number of file in the document
import glob
x = 'C:/Users/Hasee/Desktop/datamining/data/pict/*'
n = len(glob.glob(x))

#flatten 
print(len(coll))
coll[0].flatten()

pic_array = []
for i in range(n):
    img = coll[i-1]
    img_array = img.flatten()
    pic_array.append(img_array)

out_array = np.array(pic_array) # Turn list of arrays into matrix

out_array
from sklearn.decomposition import PCA
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)  
pca = PCA(n_components=5000)       
pca_result = pca.fit(out_array)

print(pca_result.explained_variance_ratio_) 
plt.plot(np.cumsum(pca_result.explained_variance_ratio_))

for i in range(n):
    mean_out_array[i] -= mean_out_array














#gray scale
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
str1 ='C:/Users/Hasee/Desktop/datamining/data/pict/lena.jpg'
coll1 = io.ImageCollection(str1)

img = np.array(Image.open('C:/Users/Hasee/Desktop/datamining/data/pict/lena.jpg').convert('L'))
io.imshow(coll1[0])
img
check(img)
#turn it into a grayscale pict
rows,cols = img.shape
for i in range(rows):
   for j in range(cols):
       if (img[i,j]<=120):
           img[i,j]=0
       else:
           img[i,j]=1
plt.figure('lena')
plt.imshow(img,cmap= 'gray')
plt.axis('off')
plt.show()


#about the label
flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png')
#接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据
#x：样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
#y：标签
#batch_size：整数，默认32
#shuffle：布尔值，是否随机打乱数据，默认为True
#save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
#save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效
#save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
#yields:形如(x,y)的tuple,x是代表图像数据的numpy数组.y是代表标签的numpy数组.该迭代器无限循环.
#seed: 整数,随机数种子

            
            
            
            
            