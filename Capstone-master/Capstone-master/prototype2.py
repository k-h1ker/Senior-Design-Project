from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
# fix random seed for reproducibility
np.random.seed(7)

# load dataset
m037_df = pd.read_csv("d1_m037.csv", sep = ",")
m037_df = m037_df.reindex(np.random.permutation(m037_df.index))

def preprocess_features(dataframe):
  """Prepares input features from California housing data set.

  Args:
    dataframe: A Pandas DataFrame expected to contain data
      from the Wine Spectral Dataset.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = dataframe[
    ["PathLength",
     "1550",
     "1560",
     "1570",
     "1580",
     "1590",
     "1600",
     "1610",
     "1620",
     "1630",
     "1640",
     "1650",
     "1660",
     "1670",
     "1680",
     "1690",
     "1700",
     "1710",
     "1720",
     "1730",
     "1740",
     "1750",
     "1760",
     "1770",
     "1780",
     "1790",
     "1800",
     "1810",
     "1820",
     "1830",
     "1840",
     "1850",
     "1860",
     "1870",
     "1880",
     "1890",
     "1900",
     "1910",
     "1920",
     "1930",
     "1940",
     "1950"]]
  return selected_features

def preprocess_targets(dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    dataframe: A Pandas DataFrame expected to contain data
      from the Wine Spectral Dataset.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  n = len(dataframe.index)

  x = pd.Series(["m037"])
  output_targets["sample"] = x.repeat(n)
  output_targets = output_targets.reset_index(drop = True)
  return output_targets

# First 28 rows in training set
training_examples = preprocess_features(m037_df.head(28)).reset_index(drop = True)
training_targets = preprocess_targets(m037_df.head(28))

# Next 8 rows in validation set
validation_examples = preprocess_features(m037_df.iloc[28:35]).reset_index(drop = True)
validation_targets = preprocess_targets(m037_df.iloc[28:35])

# Last 4 rows in test set
test_examples = preprocess_features(m037_df.tail(4)).reset_index(drop = True)
test_targets = preprocess_targets(m037_df.tail(4))

print(training_examples.shape)
print(training_targets.shape)
print(validation_examples.shape)
print(validation_targets.shape)
print(test_examples.shape)
print(test_targets.shape)

"""
X = training_examples.to_numpy(dtype = np.float64)
Y = training_targets.to_numpy(dtype = np.int8)

# create model
model = Sequential()
model.add(Dense(12, input_dim = 42, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
# Compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Fit the model
model.fit(X, Y, epochs = 150, batch_size = 10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
print(predictions)
"""
