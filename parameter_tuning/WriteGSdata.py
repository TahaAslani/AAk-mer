#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:01:43 2020

@author: taha
"""

import pickle
import numpy as np
import pandas as pd
import os
import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys

SPE = sys.argv[1]
FullName = sys.argv[2]
INPUTFOLDER = sys.argv[3]



OUTPUTFOLDER='/pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/GSdata/'

os.system('mkdir /pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE)
os.system('mkdir /pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/GSdata/')
os.system('mkdir /pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/GSinner/')
os.system('mkdir /pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/GSouter/')
os.system('mkdir /pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/BestPar/')

HeldOutAdd = '/pylon5/br5phhp/tv349/AMR/BESTPAR/'+SPE+'/HeldOut'
os.system('mkdir /pylon5/br5phhp/tv349/AMR/BESTPAR/'+SPE)
os.system('mkdir '+HeldOutAdd)

for ii in range(5):
    os.system('mkdir /pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/GSouter/Fold'+str(ii))
    os.system('mkdir /pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/BestPar/Fold'+str(ii))


# get the current path
cwd = INPUTFOLDER


# ========================================================================
df = pd.DataFrame()

FileCounter = 0
for file in sorted(os.listdir(cwd)):
    if file.endswith(".p"):
        df_temp = pickle.load(open(cwd+'/'+file, "rb"))
        df = pd.concat([df, df_temp])

        del df_temp
        gc.collect()

        FileCounter = FileCounter + 1
        print(str(FileCounter)+' file(s) loaded!')
# ========================================================================

# All indices:
INDonMETA = np.arange(df.shape[0])

MIClocation = (df.columns=='MIC').nonzero()[0][0]
Xfull = df.iloc[:,MIClocation+1:].values
Yfull = df.loc[:,'MIC'].astype('float').values

# Remove Low Frequency target values:
TargetRemoveTH = 3

Unique, Counts = np.unique(Yfull, return_counts=True)
LowRepSamples = Unique[Counts <=TargetRemoveTH]

KeepMask = np.ones(Yfull.shape).astype('bool')
for ii in range(len(Yfull)):
    if Yfull[ii] in LowRepSamples:
        KeepMask[ii] = False

Xfull = Xfull[KeepMask]
Yfull = Yfull[KeepMask]
INDonMETA = INDonMETA[KeepMask]

CLASS = np.zeros(Yfull.shape)
for ii,element in enumerate(set(Yfull)):
    CLASS[Yfull==element] = ii


Train_ind, Test_ind = train_test_split(range(Xfull.shape[0]), test_size=0.1, random_state=1, stratify = CLASS)

X = Xfull[Train_ind]
Y = Yfull[Train_ind]

pickle.dump(Train_ind, open(HeldOutAdd+"/Train_ind.p", "wb"))
pickle.dump(Test_ind, open(HeldOutAdd+"/Test_ind.p", "wb"))

pickle.dump(INDonMETA[Train_ind], open(HeldOutAdd+"/Train_ind_on_meta.p", "wb"))
pickle.dump(INDonMETA[Test_ind], open(HeldOutAdd+"/Test_ind_on_meta.p", "wb"))


# Translate Indices to names:
I2N = pickle.load(open("/pylon5/br5phhp/tv349/AMR/IndNameMap/"+FullName+"-I2N.p", "rb"))
Train_ID = [I2N[ind] for ind in INDonMETA[Train_ind]]
Test_ID = [I2N[ind] for ind in INDonMETA[Test_ind]]
pickle.dump(Train_ID, open(HeldOutAdd+"/Train_ID.p", "wb"))
pickle.dump(Test_ID, open(HeldOutAdd+"/Test_ID.p", "wb"))


#set up stratified k-fold
kfold = 5
skf = StratifiedKFold(n_splits=kfold,random_state=0,shuffle=True)

CLASScv = np.zeros(Y.shape)
for ii,element in enumerate(set(Y)):
    CLASScv[Y==element] = ii

#5-fold cross validation
for i, (train_index, test_index) in enumerate(skf.split(X, CLASScv)):
    pickle.dump(train_index, open(OUTPUTFOLDER+"/train_index-cv"+str(i)+".p", "wb"))
    pickle.dump(test_index, open(OUTPUTFOLDER+"/test_index-cv"+str(i)+".p", "wb"))
