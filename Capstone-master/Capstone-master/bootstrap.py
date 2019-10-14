import math
import numpy as np
import pandas as pd
from sklearn.utils import resample

reds = pd.read_csv('WineGroundTruth/Reds.csv')
print(reds)
names = reds['Wine ID']
num_classes = len(names)
print(num_classes)
data = pd.read_csv('D2RedWines/reds_d2_5mm.csv')

X = data.drop(['PathLength', 'Time', 'Date'], axis = 1)
#print(X)
X2 = pd.DataFrame([], columns = X.columns)

seed = 7
np.random.seed(seed)

for name in names:
    p = X[X['Sample'] == name]
    original = p.shape[0]
    if(original == 0):
        print(name)
        continue
    for i in range(71 - p.shape[0]):
        boot = resample(p, replace = True, n_samples = int(math.floor(np.random.uniform(low = 0.1) * original)), random_state = seed)
        new_s = boot.mean().to_dict()
        new_s['Sample'] = name
        X2 = X2.append(pd.Series(new_s), ignore_index = True)
print(X2)
X2.to_csv('D2RedWines/reds_d2_5mm_boot.csv', index = False)
