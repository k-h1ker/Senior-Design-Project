import os
import json
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import optimizers, utils
from keras.layers import Dense, Dropout, Activation
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from keras.models import model_from_json, load_model, Sequential
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, r2_score

reds = pd.read_csv('WineGroundTruth/Whites.csv')
#print(whites)
names = reds['Wine ID']
print(names)
num_classes = len(names)
encoder = LabelBinarizer()
#transfomed_label = encoder.fit_transform(names)
#print(transfomed_label)
'''
encoding = {}
for name in names:
    number = int(name[1:4])
    encoding[name] = number
'''
#print(encoding)

seed = 7
np.random.seed(seed)

real = pd.read_csv('WhiteWines/whites_d1_05mm.csv')
real = real.drop(['PathLength', 'Time', 'Date'], axis = 1)
boot = pd.read_csv('WhiteWines/whites_d1_05mm_boot.csv')
real2 = real.drop(['Sample'], axis = 1)
#print(boot.shape)
print(real)
'''
kfold2 = StratifiedKFold(n_splits = 2, shuffle = True, random_state = seed)
for train2, test2 in kfold2.split(real2, real.Sample):
    boot = boot.append(real.iloc[train2, :])
    break
'''

real_test = real.copy()

#print(real_test)
#print(type(real_test))
#print(boot)
#print('stop')
data = boot.copy()
#print(data)
#data.replace(encoding, inplace = True)
#print(data)
# Target Labels - Sample
#Y = encoder.fit_transform(data.Sample)
Y = data.Sample
#print(Y)
# Isolate Data - Wavelengths (nm)
#X = data.drop(['Sample', 'PathLength', 'Time', 'Date'], axis = 1) # FOR pre-bootstrap data
X = data.drop(['Sample'], axis = 1) # post-bootstrap data
#print(X)

# Should we scale the data with `StandardScaler`?
sc = StandardScaler()
X = sc.fit_transform(X)
print(type(X))
# Possible Optimizers:

#sgd = optimizers.SGD(lr = 0.0001, momentum = 0.65, decay = 0.0, nesterov = False)
"""
    Defaults: (lr = 0.01, momentum = 0.0, decay = 0.0, nesterov = False)

    lr: float >= 0. Learning rate.
    momentum: float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations.
    decay: float >= 0. Learning rate decay over each update.
    nesterov: boolean. Whether to apply Nesterov momentum.

"""

#rms_prop = optimizers.RMSprop(lr = 0.001, rho = 0.9, epsilon = None, decay = 0.0)
"""
    Defaults: (lr = 0.001, rho = 0.9, epsilon = None, decay = 0.0)

    It is recommended to leave the parameters of this optimizer at their default values (except the learning rate,
    which can be freely tuned).

    This optimizer is usually a good choice for recurrent neural networks.

    lr: float >= 0. Learning rate.
    rho: float >= 0.
    epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
    decay: float >= 0. Learning rate decay over each update.

"""

#adagrad = optimizers.Adagrad(lr = 0.01, epsilon = None, decay = 0.0)
"""
    Defaults: (lr = 0.01, epsilon = None, decay = 0.0)

    Adagrad is an optimizer with parameter-specific learning rates, which are adapted
    relative to how frequently a parameter gets updated during training. The more updates a parameter
    receives, the smaller the learning rate.

    It is recommended to leave the parameters of this optimizer at their default values.

    lr: float >= 0. Initial learning rate.
    epsilon: float >= 0. If None, defaults to K.epsilon().
    decay: float >= 0. Learning rate decay over each update.

"""

adadelta = optimizers.Adadelta(lr = 1.0, rho = 0.95, epsilon = None, decay = 0.0)
"""

    Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving
    window of gradient updates, instead of accumulating all past gradients. This way,
    Adadelta continues learning even when many updates have been done. Compared to Adagrad,
    in the original version of Adadelta you don't have to set an initial learning rate.
    In this version, initial learning rate and decay factor can be set, as in most other Keras optimizers.

    It is recommended to leave the parameters of this optimizer at their default values.

    lr: float >= 0. Initial learning rate, defaults to 1. It is recommended to leave it at the default value.
    rho: float >= 0. Adadelta decay factor, corresponding to fraction of gradient to keep at each time step.
    epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
    decay: float >= 0. Initial learning rate decay.

"""

#adam = optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
"""
    Link to Paper on Adam Optimizer: https://arxiv.org/abs/1412.6980v8

    Default parameters follow those provided in the original paper.

    lr: float >= 0. Learning rate.
    beta_1: float, 0 < beta < 1. Generally close to 1.
    beta_2: float, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
    decay: float >= 0. Learning rate decay over each update.
    amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm
    from the paper "On the Convergence of Adam and Beyond".

"""

#adamax = optimizers.Adamax(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0)
"""

    Adamax optimizer from Adam paper's Section 7.

    It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the paper.

    lr: float >= 0. Learning rate.
    beta_1: floats, 0 < beta < 1. Generally close to 1.
    beta_2: floats, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
    decay: float >= 0. Learning rate decay over each update.

"""

#nadam = optimizers.Nadam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, schedule_decay = 0.004)
"""

    Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper. It is recommended to leave the
    parameters of this optimizer at their default values.

    lr: float >= 0. Learning rate.
    beta_1: floats, 0 < beta < 1. Generally close to 1.
    beta_2: floats, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
    schedule_decay: floats, 0 < schedule_decay < 1.

"""

"""
Hint: Want MSE and MAE close to zero, R2 close to 1

  Performance with Default Values:
    sgd: MSE: 143.96900939941406; MAE: 9.732954025268555; R2: -0.26427227510446105;
    rms_prop: MSE: 168.92559814453125; MAE: 10.763418197631836; R2: -0.4834301784694597;
    adagrad: MSE: 117.603271484375; MAE: 9.108181953430176; R2: -0.032740072707617696;
    adadelta: MSE: 118.69427490234375; MAE: 9.107599258422852; R2: -0.042320704822709665;
    adam: MSE: 129.95680236816406; MAE: 8.755480766296387; R2: -0.1412233089628927;
    adamax: MSE: 281.8838195800781; MAE: 14.188810348510742; R2: -1.4753792288146257;
    nadam: MSE: 114.07797241210938; MAE: 8.953316688537598; R2: -0.001782526418631969;
  with Structure:
    model = Sequential()
    model.add(Dense(82, input_dim = 41, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1))


    model.add(Dense(410, input_dim = 41, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(1))

    model.add(Dense(410, input_dim = 41, activation = 'tanh'))
    model.add(Dense(128, activation = 'tanh'))
    model.add(Dense(64, activation = 'tanh'))
    model.add(Dense(1))

"""
'''
for j in range(150, 325, 25):
    print(j)
    Y = data.Sample
    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    for train, test in kfold.split(X, Y):
        #encoder.fit_transform(data.Sample)
        Y = encoder.fit_transform(data.Sample)
        model = Sequential()
        model.add(Dense(410, input_dim = 41, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(num_classes))
        model.add(Activation('selu'))
        model.compile(optimizer = adadelta, loss = 'mse', metrics = ['mae'])
        model.fit(X[train], Y[train], epochs = j, verbose = 0)
'''
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
for train, test in kfold.split(X, Y):
    #encoder.fit_transform(data.Sample)
    Y = encoder.fit_transform(data.Sample)
    model = Sequential()
    model.add(Dense(410, input_dim = 41, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(num_classes))
    model.add(Activation('selu'))
    model.compile(optimizer = adadelta, loss = 'mse', metrics = ['mae'])
    model.fit(X[train], Y[train], epochs = 1000, verbose = 0)

#test = [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71, 80, 81, 90, 91, 100, 101, 110, 111, 120, 121, 130, 131, 140, 141, 150, 151]
# real_test
real_test2 = real_test.drop(['Sample'], axis = 1)


y_pred = encoder.inverse_transform(model.predict(sc.transform(real_test2.iloc[5:, :])))
sum = 0
temp = real_test.Sample.values
for i in range(len(y_pred)):
    if (y_pred[i] == temp[i]) == True:
        sum += 1
print((float(sum) / float(len(y_pred))))
from sklearn.utils import resample
acc = [(float(sum) / float(len(y_pred)))]
#y_pred = encoder.inverse_transform(model.predict(X[test]))
for k in range(10):
    original = real_test2.shape[0]
    real_test3 = resample(real_test2, replace = True, n_samples = original, random_state = seed)
    y_pred = encoder.inverse_transform(model.predict(sc.transform(real_test3)))
    sum = 0
    temp = real_test.Sample.values
    for i in range(len(y_pred)):
        if (y_pred[i] == temp[i]) == True:
            sum += 1

    acc.append((float(sum) / float(len(y_pred))))

print(np.mean(acc))
print(np.std(acc))

"""
# Option 1: Save weights and model architecture separately

# Save the weights
model.save_weights('WhiteWines/model_weights.h5')

# Save the model architecture
with open('WhiteWines/model_architecture.json', 'w') as f:
    f.write(model.to_json())
"""

"""
# Model reconstruction from JSON file
with open('WhiteWines/model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('WhiteWines/model_weights.h5')
"""
'''
'''
# Option 2: Save whole model (weights + architecture) in one file

import pickle

list_pickle_path = 'D2RedWines/reds_d2_5mm.pkl'

# Create an variable to pickle and open it in write mode
list_pickle = open(list_pickle_path, 'wb')
pickle.dump(encoder, list_pickle)
list_pickle.close()

del encoder

# Creates a HDF5 file 'my_model.h5'
model.save('D2RedWines/reds_d2_5mm.h5')

# Deletes the existing model
del model

# Returns a compiled model identical to the previous one

list_unpickle = open(list_pickle_path, 'r')
encoder = pickle.load(list_unpickle)
model = load_model('D2RedWines/reds_d2_5mm.h5')

import time

start = time.time()

y_pred = encoder.inverse_transform(model.predict(sc.transform(real_test2)))

end = time.time()

print(end-start)
'''
