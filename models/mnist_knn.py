
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:45:42 2017

@author: wangxinji
"""


import timeit
start = timeit.default_timer()
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import os
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn import preprocessing,cross_validation,metrics
import numpy as np

#change working directory
os.chdir('C:\\Users\\Hasee\\Desktop')
#read data 
train = pd.read_csv("train.csv", sep=",", error_bad_lines=False)
test = pd.read_csv("pcatr.csv", sep=",")
train
#append data 
dataframe = train
print("training data points: {}".format(len(train)))
# Extract values
data = dataframe.values
features_unscaled = data[:, 1:]
output = data[:, 0]

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
x = preprocessing.scale(x) #standardlize data feature
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.2)

#instantiate learning moedel(k = 10)
knn = KNeighborsClassifier(n_neighbors = 10)
#fitting the model 
print("start traning! please wait")
knn.fit(x_train,y_train)

#evaluate accuracy
accuracy = knn.score(x_test,y_test)
print('your model accuarcy is: ',accuracy,'%(with n = 10) ')
stop = timeit.default_timer()
print("Run time: " + str(int(stop - start)) + " seconds.")

#model evaluate
y_expect = y_test
start1 = timeit.default_timer()
y_prdict = knn.predict(x_test)
print(metrics.classification_report(y_expect,y_prdict))
cv_results = cross_validate(knn,test.loc[:, test.columns != 'labels'], test['labels'])
acc_res = np.mean(cv_results['test_score'])
print('accuracy is that :',acc_res)
stop1 = timeit.default_timer()
print("Run time: " + str(int(stop1 - start1)) + " seconds.")






























