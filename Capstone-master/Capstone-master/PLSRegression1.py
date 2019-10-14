#First attempt at building a PLS regression model (march 5th 1230AM)
#!/usr/bin/python
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

#pls = PLSRegression(n_components =  5)      #figure out what this means and optimize it

#pls.fit(x_calib, y_calib)                     #we're gunna have to import the data somehow - see if fareen can send us the cleaned up data if he hasnt already

#y_pred = pls.predict(x_valid)                   #calib = calibration (training) set
                                                #valid = validation (test) set
#score = r2_score(y_valid, y_pred)

#mse = mean_squared_error(y_valid, y_pred)



#Edit March 9th

#data import and preprocessing
data = pd.read_csv('Detector_1.csv')
#print data

reference_data = pd.DataFrame(data['Sample'])
#print reference_data

y_c= reference_data[:1145]      #doing an 80%-20% split on the data so the first 1145 data point (rows) will be training and the remaining 286 will be test
y_v = reference_data[1146:]
y_calib = pd.to_numeric(y_c['Sample'].str[14:17], errors='coerce')
y_valid = pd.to_numeric(y_v['Sample'].str[14:17], errors='coerce')

#print y_calib


x_calib = pd.DataFrame(data.iloc[:1145, 5:])   #insert data ranges in brackets
x_valid = pd.DataFrame(data.iloc[1146:, 5:])
#print y_calib['Sample'].str[14:17]

wl = np.array(list(data)[5:])

plt.figure(figsize=(8,4.5))
with plt.style.context(('ggplot')):
    plt.plot(wl, x_calib.T)
    plt.xlabel('Wavelength')
    plt.ylabel('Absorbance')
    plt.tick_params(axis = 'x', rotation = 45)
plt.show(block=False)
_ = raw_input("Press [enter] to continue.")
#Edits added 3/13

x2_calib = savgol_filter(x_calib, 17, polyorder = 2, deriv = 2)
x2_valid = savgol_filter(x_valid, 17, polyorder = 2, deriv = 2)

plt.figure(figsize=(8,4.5))
with plt.style.context(('ggplot')):
    plt.plot(wl, x2_calib.T)
    plt.xlabel('Wavelength')
    plt.ylabel('D2 Absorbance')
    plt.tick_params(axis = 'x', rotation = 45)
plt.show(block = False)
_ = raw_input("Press [enter] to continue.")

def prediction(x_calib, y_calib, x_valid, y_valid, plot_components=False):
    mse = []
    component = np.arange(1,40)
    for i in component:
            pls = PLSRegression(n_components = i)
            pls.fit(x_calib, y_calib)
            y_pred = pls.predict(x_valid)

            mse_p = mean_squared_error(y_valid, y_pred)
            mse.append(mse_p)

            comp = 100*(i+1)/40
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
    y_pred = pls.predict(x_valid)

    score_p = r2_score(y_valid, y_pred)
    mse_p = mean_squared_error(y_valid, y_pred)
    sep = np.std(y_pred[:,0]-y_valid)
    rpd = np.std(y_valid)/sep
    bias = np.mean(y_pred[:,0]-y_valid)

    print('R2: %5.3f'  % score_p)
    print('MSE: %5.3f' % mse_p)
    print('SEP: %5.3f' % sep)
    print('RPD: %5.3f' % rpd)
    print('Bias: %5.3f' %  bias)

    rangey = max(y_valid) - min(y_valid)
    rangex = max(y_pred) - min(y_pred)

    z = np.polyfit(y_valid, y_pred, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_pred, y_valid, c='red', edgecolors='k')
        ax.plot(z[1]+z[0]*y_valid, y_valid, c='blue', linewidth=1)
        ax.plot(y_valid, y_valid, color='green', linewidth=1)
        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        plt.title('Prediction')

        plt.text(min(y_pred)+0.05*rangex, max(y_valid)-0.1*rangey, 'R$^{2}=$ %5.3f'  % score_p)
        plt.text(min(y_pred)+0.05*rangex, max(y_valid)-0.15*rangey, 'MSE: %5.3f' % mse_p)
        plt.text(min(y_pred)+0.05*rangex, max(y_valid)-0.2*rangey, 'SEP: %5.3f' % sep)
        plt.text(min(y_pred)+0.05*rangex, max(y_valid)-0.25*rangey, 'RPD: %5.3f' % rpd)
        plt.text(min(y_pred)+0.05*rangex, max(y_valid)-0.3*rangey, 'Bias: %5.3f' %  bias)
        plt.show(block = False)
        _ = raw_input("Press [enter] to continue.")
prediction(x2_calib, y_calib, x2_valid, y_valid, plot_components=True)
