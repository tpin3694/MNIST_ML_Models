# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 00:24:07 2017

@author: wangxinji
"""

import timeit
import os
start = timeit.default_timer()
os.chdir('C:\\Users\\Hasee\\Desktop\\datamining\\data')
# Load remaining libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing,cross_validation,metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


#define pca method
def zeroMean(x):
    meanVal = np.mean(x,axis = 0)#axis = 0 mean calc mean by ROW
    newData = x-meanVal
    return newData,meanVal

def percentage2n(eigVals,percentage):  
    sortArray=np.sort(eigVals)   #ascengding   
    sortArray=sortArray[-1::-1]  #descending  
    arraySum=sum(sortArray)  
    tmpSum=0  
    num=0  
    for i in sortArray:  
        tmpSum+=i  
        num+=1  
        if tmpSum>=arraySum*percentage:  
            return num  



def pca_fit(data,percentage):
    zeroMean(data)
    newData,meanVal=zeroMean(data) 
    covMat=np.cov(newData,rowvar=0)  
    eigvals,eigVects=np.linalg.eig(np.mat(covMat))  
    n=percentage2n(eigvals,percentage)
    eigValIndice=np.argsort(eigvals)            #ascengding  
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   
    n_eigVect=eigVects[:,n_eigValIndice]        #the first n eigValue   
    lowDDataMat=newData*n_eigVect               #dataset in low demension 
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #the reconstitution data
    return lowDDataMat 


# Create model
def ini_mod():
    # Define Model type
    model = Sequential()
    # Define Layers
    model.add(Dense(16, input_dim=784, kernel_initializer = "random_normal",
              bias_initializer = "zeros", activation="sigmoid"))
    model.add(Dense(32, kernel_initializer = "random_normal",
              bias_initializer = "zeros", activation="relu"))
    model.add(Dense(10, kernel_initializer = "random_normal",
              bias_initializer = "zeros", activation="softmax"))
    
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#change working directory
os.chdir('C:\\Users\\Hasee\\Desktop\\datamining\\data')

# Set the seed
seed = 28
np.random.seed(seed)


#read data 
train = pd.read_csv("train.csv", sep=",",)
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
dummy_out.reshape(-1, 1)

# Scale features to -1 to 1
scaler = MinMaxScaler(feature_range=(-1, 1))
features = scaler.fit_transform(features_unscaled)
x = features
y = output
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.2)
y_train = y_train.reshape((-1, 1))
#this is a simpler model
model = Sequential()
model.add(Dense(32, activation='sigmoid', input_dim=784))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

dummy_out.reshape(-1, 1)
X_train = features
Y_train = dummy_out
x_train = features   
y_train = dummy_out

start2 = timeit.default_timer()
print('=============start traning!=====================')
model.fit(X_train,Y_train, epochs=10, batch_size=32)
print('==========================traning over=========================')
end2 = timeit.default_timer()
print("nerual net model Run time: " + str(int(end2 - start2)) + " seconds.")
y_pred = model.predict(x_test)
y_pred.shape
y_test.shape
y_pred[:1]
y_test[1,1]


start1 = timeit.default_timer()
print('================start traning new model============== ')
x = pca_fit(features,0.9)

x.shape
#new model
model = Sequential()
model.add(Dense(32, activation='sigmoid', input_dim=87))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#traning new model
model.fit(x,dummy_out, epochs=10, batch_size=32)
print('=====================end traning new model================== ')
end1 = timeit.default_timer()
print("pca nerual net model Run time: " + str(int(end1 - start1)) + " seconds.")
y_pred = model.predict(x)
y_pred
m ,n= y_pred.shape
m
y_prediction = []
for i in range(y_pred.shape[1,]):
    y_prediction[i] = np.argmax(y_pred[i:])

stop = timeit.default_timer()
#results
print("total Run time: " + str(int(stop - start)) + " seconds.")























