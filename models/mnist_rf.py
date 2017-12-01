import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt


usr_dir = "/home/tpin3694/Documents/university/MSc/dataMining/group_work/emotion_recognition/data/mnist/split/"


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

train = data_load(usr_dir, "train")
train_features = train.values[:, 1:].astype(int)
train_target = train.values[:, 0].astype(int)
recogniser = RandomForestClassifier(250)
#score = cross_val_score(recogniser, train_features, train_target, cv = 10)
#mean_acc, sd_acc = np.mean(score), np.std(score)
#score = cross_val
test = data_load(usr_dir, "test")
test_features = test.values[:, 1:].astype(int)
test_target = test.values[:, 0].astype(int)
recogniser.fit(train_features, train_target)
recogniser.predict(test_features)
preds = test_target[recogniser.predict(test_features)]
pd.crosstab(test_target, preds, rownames = ["Actual Value"], colnames = ["Predicted Value"])
cv_results = cross_validate(recogniser, test_features, test_target, cv = 10) #, scoring=("r2", "accuracy", "average_precision", "roc_auc"))
acc_res = np.mean(cv_results['test_score'])
print("Accuracy: " + str(acc_res))

# def main():
#     train = data_load(usr_dir, "train")
#     train_features = train.values[:, 1:].astype(int)
#     train_target = train.values[:, 0].astype(int)
#     tree_counter = np.linspace(10, 100, 20, dtype=int)
#     mean_scores, sd_scores = [], []
#     for tree in tree_counter:
#         print("Processing Tree Number " + str(tree) + ".")
#         recogniser = RandomForestClassifier(tree)
#         score = cross_val_score(recogniser, train_features, train_target)
#         mean_scores.append(np.mean(score))
#         sd_scores.append(np.std(score))
#     mean_array = np.array(mean_scores)
#     std_array = np.array(sd_scores)
#     plotter(mean_scores, mean_array, std_array, tree_counter, usr_dir)
#     print("Maximum Accuracy Achieved: " + str(max(mean_scores)))
#
# if __name__ == "__main__":
#     main()
