#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:03:30 2020

@author: taha
"""


print('Importing packages')
import pandas as pd
import numpy as np
import scipy.stats
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.datasets import load_svmlight_file
import os
import sys
import csv
from collections import defaultdict
import pickle
import gc
print('Packages imported!')


SPE = sys.argv[1]


NumberofFolds = 5


PARFOLDER = '/pylon5/br5phhp/tv349/AMR/BESTPAR/'+SPE
os.system('mkdir '+PARFOLDER)

DF = pd.DataFrame(index=range(NumberofFolds),columns=[\
                                                      'Test Acc','Test RMSE' ,'eta', 'max_depth',\
                                                          'min_child_weight', 'lambda', 'gamma', 'colsample_bytree', 'max_delta_step','n_estimators'])

BestError = np.inf
for FoldNum in range(NumberofFolds):

    study = pickle.load(open('/pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/BestPar/Fold'+str(FoldNum)+'/STUDY.p', "rb"))
    BestPar = study.best_trial.params
    pickle.dump(BestPar, open(PARFOLDER+'/Fold'+str(FoldNum)+'.p', "wb"))

    for key in BestPar.keys():
        DF.loc[FoldNum,key] = BestPar[key]
    
    # Find best fold
    OuterFolder='/pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/GSouter/Fold'+str(FoldNum)
    RMSE = pickle.load(open(OuterFolder+"/RMSE-cv"+str(FoldNum)+".p", "rb"))
    acc = pickle.load(open(OuterFolder+"/acc-cv"+str(FoldNum)+".p", "rb"))
    
    DF.loc[FoldNum,'Test Acc'] = acc
    DF.loc[FoldNum,'Test RMSE'] = RMSE
    
    if RMSE<BestError:
        OPTIMAL = BestPar
        BestError = RMSE
        LeastErrorFold = FoldNum

pickle.dump(OPTIMAL, open(PARFOLDER+'/LeastErrorPar.p', "wb"))
pickle.dump(LeastErrorFold, open(PARFOLDER+'/LeastErrorFoldInd.p', "wb"))

for col in DF.columns:
    DF.loc[6,col] = np.mean(DF.loc[:,col])
DF.loc[6,'Test Acc'] = np.nan
DF.loc[6,'Test RMSE'] = np.nan
DF.loc[6,'max_depth'] = int(np.round(DF.loc[6,'max_depth']))
DF.loc[6,'n_estimators'] = int(np.round(DF.loc[6,'n_estimators']))
DF.to_csv('/pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/GSouter/Summary.tsv',sep='\t')

AVE = {}
for col in DF.columns:
    if col == 'Test Acc':
        continue
    if col == 'Test RMSE':
        continue
    AVE[col] = DF.loc[6,col]

pickle.dump(AVE, open(PARFOLDER+'/AVE.p', "wb"))

print('All Done!')
