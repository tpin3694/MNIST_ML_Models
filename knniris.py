# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:47:12 2017

@author: wangxinji
"""
import timeit
start = timeit.default_timer()
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
from keras.utils import np_utils
import pandas as pd
from sklearn import preprocessing,cross_validation,neighbors

#change working directory
os.chdir('C:\\Users\\Hasee\\Desktop\\datamining\\data')
#read data 
train = pd.read_csv("iris_train.txt", sep=",", header = None)
test = pd.read_csv("iris_test.txt", sep=",", header = None)
#append data 
dataframe = train.append(test)

# Extract values
data = dataframe.values
features_unscaled = data[:, 0:4]
output = data[:, 4]

# Encode output label - Hot Pot Encoding
encoder = LabelEncoder()
encoder.fit(output)
encoded_out = encoder.transform(output)
dummy_out = np_utils.to_categorical(encoded_out)

# Scale features to -1 to 1
scaler = MinMaxScaler(feature_range=(-1, 1))
features = scaler.fit_transform(features_unscaled)
x = features
y = output
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.2)

#instantiate learning moedel(k = 5)
knn = KNeighborsClassifier(n_neighbors = 5)
#fitting the model 
knn.fit(x_train,y_train)

#evaluate accuracy
accuracy = knn.score(x_test,y_test)
print(accuracy)

#input example and predict
example = np.array([2,6,12,4])
example = example.reshape(1,-1)
pred = knn.predict(example)
print('predict of example is :',pred)



stop = timeit.default_timer()

print("Run time: " + str(int(stop - start)) + " seconds.")
