import numpy as np
import pandas as pd

whites = ['WineGroundTruth/Whites_Flight1.csv',
          'WineGroundTruth/Whites_Flight2.csv',
          'WineGroundTruth/Whites_Flight3.csv']

reds = ['WineGroundTruth/Reds_Flight4.csv',
        'WineGroundTruth/Reds_Flight5.csv',
        'WineGroundTruth/Reds_Flight6.csv',
        'WineGroundTruth/Reds_Flight7.csv']

whites_df1 = pd.read_csv(whites[0])
whites_df2 = pd.read_csv(whites[1])
whites_df3 = pd.read_csv(whites[2])

reds_df4 = pd.read_csv(reds[0])
reds_df5 = pd.read_csv(reds[1])
reds_df6 = pd.read_csv(reds[2])
reds_df7 = pd.read_csv(reds[3])

wframes = [whites_df1, whites_df2, whites_df3]
wresult = pd.concat(wframes).reset_index(drop = True)
wfilename = 'WineGroundTruth/Whites.csv'
wresult.to_csv(wfilename, index = False)

rframes = [reds_df4, reds_df5, reds_df6, reds_df7]
rresult = pd.concat(rframes).reset_index(drop = True)
rfilename = 'WineGroundTruth/Reds.csv'
rresult.to_csv(rfilename, index = False)
