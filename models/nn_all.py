# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:02:05 2017

@author: wangxinji
"""

import timeit
import os
from sklearn.model_selection import StratifiedKFold

start = timeit.default_timer()
os.chdir('C:\\Users\\Hasee\\Desktop\\datamining\\data\\NN')
# Load remaining libraries
import pandas as pd
from sklearn.model_selection import cross_validate
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing,cross_validation,metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

def nn_tuner(optim_func, num_cols):
    model = Sequential()
    model.add(Dense(32, activation='sigmoid', input_dim=num_cols))
    model.add(Dense(10, activation="tanh"))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optim_func, loss='binary_crossentropy',metrics=['accuracy'])
    return model


# Read in data
train = pd.read_csv("train_3d.csv", sep=",",)
test = pd.read_csv('test_3d.csv',sep = ',')

# Get number of features
num_cols = train.shape[1]-1

# Split out features and labels
train_x = train.values[:, 1:].astype(float)
train_y = train.values[:, 0].astype(int)
test_x = test.values[:, 1:].astype(float)
test_y = test.values[:, 0].astype(int)

# One-Hot Encode 
train_y_enc = np_utils.to_categorical(train_y)
test_y_enc = np_utils.to_categorical(test_y)
class_count = test_y_enc.shape[1]

# Setup Cross-Validation
# Change CV to 10
kfold = KFold(n_splits=3, random_state=123)
cross_scores = []
for train_ind, test_ind in kfold.split(test):
    new_model = nn_tuner("adam", num_cols)
    # Change Epochs Xinji
    new_model.fit(train_x[train_ind], train_y_enc[train_ind], epochs = 5, batch_size = 10, verbose = 1)
    print(new_model.evaluate(test_x[test_ind], test_y_enc[test_ind], verbose = 0))
    
# Assess final model
final_model = nn_tuner("adam", num_cols)
predict_y = pd.DataFrame(final_model.predict(test_x))

predicted_numbers = []
for i in range(predict_y.shape[0]):
    to_assess = list(predict_y.iloc[i])
    predicted_numbers.append(to_assess.index(max(to_assess)))
    
truth_list = test_y.tolist()
results = pd.DataFrame(
        {"Prediction": predicted_numbers,
         "Truth": truth_list
         })
    
def accuracy(x):
    if x["Prediction"] == x["Truth"]:
        return 1
    else:
        return 0

results["results"] = results.apply(accuracy, 1)

print("Model Accuracy: ", str(sum(results["results"]/results.shape[0])))
