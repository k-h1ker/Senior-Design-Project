from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

#imports added on 3/9
from sys import stdout
import numpy as np              #David, you will need these python libraries. Numpy, Pandas, Scipy, and sklearn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer   #We probably wont need these libraries
from sklearn.metrics import accuracy_score
import time

data_training = pd.read_csv('whites_d2_1mm_boot.csv')  #we train on the boot strapped data
data_test = pd.read_csv('whites_d2_1mm.csv')           #we test on the actual

reference_data_training = pd.DataFrame(data_training['Sample'])
reference_data_test = pd.DataFrame(data_test['Sample'])

y_pre_training = reference_data_training[:]
y_pre_test = reference_data_test[:]

y_calib = pd.to_numeric(y_pre_training['Sample'].str[1:4], errors='coerce')
y_valid = pd.to_numeric(y_pre_test['Sample'].str[1:4], errors='coerce')

x_calib = pd.DataFrame(data_training.iloc[:, :-1])
x_valid = pd.DataFrame(data_test.iloc[:, 3:-1])
lb = preprocessing.LabelBinarizer()

y_calib = lb.fit_transform(y_calib)
y_valid = lb.fit_transform(y_valid)


x2_calib = savgol_filter(x_calib, 17, polyorder = 2, deriv = 2)
x2_valid = savgol_filter(x_valid, 17, polyorder = 2, deriv = 2)

def prediction(x_calib, y_calib, x_valid, y_valid, plot_components=False):
    mse = []
    component = np.arange(1,30)
    for i in component:
            pls = PLSRegression(n_components = i)
            pls.fit(x_calib, y_calib)
            y_pred = pls.predict(x_valid)

            mse_p = mean_squared_error(y_valid, y_pred)
            mse.append(mse_p)

            comp = 100*(i+1)/30
            stdout.write("\r%d%% completed" %comp)
            stdout.flush()
    stdout.write("\n")

    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin+1)
    stdout.write("\n")

    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc = 'blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc = 'red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(xmin=-1)

        plt.show(block = False)
        _ = raw_input("Press [enter] to continue.")
    pls = PLSRegression(n_components = msemin+1)
    pls.fit(x_calib, y_calib)

    startTime = time.time()
    y_pred = pls.predict(x_valid)
    endTime = time.time()
    print('Time elapsed: %s seconds' % (endTime - startTime))

    lb = preprocessing.LabelBinarizer()
    score_p = r2_score(y_valid, y_pred)
    mse_p = mean_squared_error(y_valid, y_pred)

    lb.fit_transform(y_valid)

    score = r2_score(y_valid, y_pred)
    print('R2: %5.3f'  % score_p)
    print('MSE: %5.3f' % mse_p)

    #print

    pr = lb.inverse_transform(y_pred)
    ac = lb.inverse_transform(y_valid)

    #print type(pr[0])
    #print ac

    sum = 0
    for j in range(len(pr)):
        if np.array_equal(pr[j], ac[j]):
            sum += 1
    print('Accuracy: ' + str((float(sum) / float(len(pr))) * 100) + '%')

    #f = open("whites_d2_1mm_model.txt","w+")
    #m = pickle.dumps(pls)
    #f.write(m)

prediction(x2_calib, y_calib, x2_valid, y_valid, plot_components=False)
