#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 08:10:56 2017

@author: tpin3694
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_validate, cross_val_score
import matplotlib.pyplot as plt
import glob
import re
import scikitplot as skplt
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import Normalize

usr_dir = "/home/tpin3694/Documents/university/MSc/dataMining/group_work/emotion_recognition/data/mnist/dim_reduced/"


def data_load(directory, data_type):
    if data_type == "test" or data_type == "train":
        print("Loading " + str(data_type) + "ing data...")
        data = pd.read_csv(directory + str(data_type) + ".csv")
        print(str(data_type).capitalize() + "ing data successfully loaded!")
        return data
    else:
        print("Please select either test or train.")
        return None


def plotter(scores, array1, array2, tree_list, dir):
    plt.plot(tree_list, scores)
    plt.plot(tree_list, array1 + array2, 'b--')
    plt.plot(tree_list, array1 - array2, 'b--')
    plt.ylabel('CV score')
    plt.xlabel('# of trees')
    plt.savefig(dir + 'plots/tsne_cv_trees.png')


def data_scorer(model, features, labels, folds):
    recall = np.mean(cross_val_score(model, X = features, y = labels, cv = folds, scoring = "recall", n_jobs = -1))
    precision = np.mean(cross_val_score(model, X = features, y = labels, cv = folds, scoring = "precision", n_jobs = -1))
    accuracy = np.mean(cross_val_score(model, X = features, y = labels, cv = folds, scoring = "accuracy", n_jobs = -1))
    f1 = np.mean(cross_val_score(model, X = features, y = labels, cv = folds, scoring = "f1", n_jobs = -1))
    auc = np.mean(cross_val_score(model, X = features, y = labels, cv = folds, scoring = "roc_auc", n_jobs = -1))
    return accuracy, recall, precision, f1, auc 


def scorer(model, features, labels):
    accuracy = cross_val_score(model, features, labels)
    mean_metric = np.mean(accuracy)
    sd_metric = np.std(accuracy)
    return mean_metric, sd_metric


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_param_space_scores(scores, C_range, gamma_range):
    """
    Draw heatmap of the validation accuracy as a function of gamma and C
    
    
    Parameters
    ----------
    scores - 2D numpy array with accuracies
    
    """
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.
    
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.jet,
               norm=MidpointNormalize(vmin=0.5, midpoint=0.9))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
plt.show()

trains = glob.glob(usr_dir+"trains/*.csv")

results_list =[]


test_train = pd.read_csv(trains[0])
test_test = pd.read_csv(usr_dir+"tests/2d.csv")
train_features = test_train.values[:, 1:].astype(int)
train_target = test_train.values[:, 0].astype(int)
test_features = test_test.values[:, 1:].astype(int)
test_target = test_test.values[:, 0].astype(int)


gam_range = np.outer(np.logspace(-4,0,4), np.array([1,5])).flatten()
C_range = np.outer(np.logspace(-1, 1, 3),np.array([1,5])).flatten() # Amount of allowed dimension wiggle
parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gam_range}
svm_clsf = svm.SVC()
grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=1, verbose=2)

import datetime as dt
start_time = dt.datetime.now()
print("Starting Parameter Space Search at {}".format(str(start_time)))
grid_clsf.fit(train_features, train_target)
elapsed = dt.datetime.now()-start_time
print("Time Taken: {}".format(str(elapsed)))

classifier = grid_clsf.best_estimator_
params = grid_clsf.best_params_

scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range), len(gam_range))
plot_param_space_scores(scores, C_range, gam_range)

from sklearn import metrics

expected = test_target
predicted = classifier.predict(test_features)

print("Accuracy: {}".format(metrics.accuracy_score(expected, predicted)))

for file in trains:
    file_type = re.sub(usr_dir+"trains/", "", file)
    if file_type == "pca.csv":
        pass
    else:          
        # Read in data
        train = pd.read_csv(file)
        print(file_type + " training read in")
        test = pd.read_csv(usr_dir+"tests/"+file_type)
        print(file_type + " testing read in")
        
        # Split out features and labels
        train_features = train.values[:, 1:].astype(int)
        train_target = train.values[:, 0].astype(int)
        test_features = test.values[:, 1:].astype(int)
        test_target = test.values[:, 0].astype(int)
        print("Data split into test/train")
        
        # Run Classifier
        recogniser = RandomForestClassifier(200, random_state = 42)
        
        # Fit model
        recogniser.fit(train_features, train_target)
        recogniser.predict(test_features)
        print("Model Fitted")
        
        # Get Model Metrics
        mean_score, sd_score = scorer(recogniser, test_features, test_target)
        results = [file_type, mean_score, sd_score]
        preds = test_target[recogniser.predict(test_features)]
        print("Model Metrics Found")
        
        # Calculate probabilities
        y_prob = recogniser.predict_proba(test_features)
        y_pred = recogniser.predict(test_features)
        
        # Append results
        results_list.append(results)

results_df = pd.DataFrame(results_list, columns = ["Dataset","Accuracy", "Standard Deviation"])
results_df.to_csv("/home/tpin3694/Documents/university/MSc/dataMining/group_work/emotion_recognition/svm_metrics.csv")