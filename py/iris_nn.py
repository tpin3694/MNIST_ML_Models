# Runtime
import timeit
start = timeit.default_timer()

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
import matplotlib.pyplot as plt

# Set the seed
seed = 2
np.random.seed(seed)

# Read data
train = pd.read_csv("data/iris_train.txt", sep=",", header = None)
test = pd.read_csv("data/iris_test.txt", sep=",", header = None)
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


# Create model
def ini_mod():
    # Define Model type
    model = Sequential()
    # Define Layers
    model.add(Dense(16, input_dim=4, kernel_initializer = "random_normal",
              bias_initializer = "zeros", activation="sigmoid"))
    model.add(Dense(32, kernel_initializer = "random_normal",
              bias_initializer = "zeros", activation="relu"))
    model.add(Dense(3, kernel_initializer = "random_normal",
              bias_initializer = "zeros", activation="softmax"))
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Fit model
# Epochs = no. iterations
# batch_size: higher (~32) = faster train, lower(~4) = slower, more accurate
estimator = KerasClassifier(build_fn=ini_mod, epochs=10, batch_size=4, verbose=0)

# Evaluate model using cross-validation 10 times
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

# Print results
results = cross_val_score(estimator, features, dummy_out, cv = kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
stop = timeit.default_timer()

print("Run time: " + str(int(stop - start)) + " seconds.")