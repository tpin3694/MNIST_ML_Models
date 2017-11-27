import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt


usr_dir = "/home/tpin3694/Documents/university/MSc/dataMining/group_work/emotion_recognition/data/mnist/split/"
def data_load(directory, data_type):
    if data_type == "test" or data_type == "train":
        print("Loading " + str(data_type) + "ing data...")
        data = pd.read_csv(usr_dir + str(data_type) + ".csv")
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
    plt.savefig(dir + '/plots/cv_trees.png')

def main():
    train = data_load(usr_dir, "train")
    train_features = train.values[:, 1:].astype(int)
    train_target = train.values[:, 0].astype(int)
    tree_counter = np.linspace(10, 80, 20, dtype= int)
    mean_scores, sd_scores = [], []
    for tree in tree_counter:
        print("Processing Tree Number " + str(tree) + ".")
        recogniser = RandomForestClassifier(tree)
        score = cross_val_score(recogniser, train_features, train_target)
        mean_scores.append(np.mean(score))
        sd_scores.append(np.std(score))
    mean_array = np.array(mean_scores)
    std_array = np.array(sd_scores)
    plotter(mean_scores, mean_array, std_array, tree_counter, usr_dir)


if __name__ == "__main__":
    main()