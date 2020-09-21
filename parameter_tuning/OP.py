#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 02:48:43 2020

@author: taha
"""

"""
Optuna example that demonstrates a pruner for XGBoost.
In this example, we optimize the validation accuracy of cancer detection using XGBoost.
We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.
"""

import numpy as np
import sklearn.datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import os
import gc
import pickle
import optuna
import sys


INPUT = sys.argv[1]
SPE = sys.argv[2]
FOLDn = sys.argv[3] 
CPUCOUNT = int(sys.argv[4])



HeldOutAdd = '/pylon5/br5phhp/tv349/AMR/BESTPAR/'+SPE+'/HeldOut'
CVindFOLDER='/pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/GSdata'
bestParFOLDER='/pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/BestPar/Fold'+str(FOLDn)
OuterFolder='/pylon5/br5phhp/tv349/AMR/GS/OPTUNA/'+SPE+'/GSouter/Fold'+str(FOLDn)

# calculate the accuracy based on the ±2 dilution
def accuracy_dilute(prdc,target):
    total = len(prdc);
    n = 0
    for i,j in zip(prdc,target):
        if j-1 <= i and i <= j+1:
            n += 1
    return n/total


# ========================================================================
cwd = INPUT
df = pd.DataFrame()

FileCounter = 0    
for file in os.listdir(cwd):
    if file.endswith(".p"):
        df_temp = pickle.load(open(cwd+'/'+file, "rb"))
        df = pd.concat([df, df_temp])

        del df_temp
        gc.collect()
        
        FileCounter = FileCounter + 1
        print(str(FileCounter)+' file(s) loaded!')

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

# load the held out set index
Train_ind = pickle.load(open(HeldOutAdd+"/Train_ind.p", "rb"))


X = Xfull[Train_ind]
Y = Yfull[Train_ind]

# load the fold index
train_index = pickle.load(open(CVindFOLDER+"/train_index-cv"+str(FOLDn)+".p", "rb"))
test_index = pickle.load(open(CVindFOLDER+"/test_index-cv"+str(FOLDn)+".p", "rb"))

train_x = X[train_index]
train_y = Y[train_index]

valid_x = X[test_index]
valid_y = Y[test_index]

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):

    #train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.1)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "silent": 1,
        "objective":"reg:squarederror",
        "eval_metric": "rmse",
        "eta": trial.suggest_loguniform("eta", 1e-5, 0.5),
        "max_depth": trial.suggest_int("max_depth", 3,10),
        "min_child_weight": trial.suggest_uniform("min_child_weight", 2,8),
        "lambda": trial.suggest_uniform("lambda", 0, 10),
        "gamma": trial.suggest_uniform("gamma", 0, 3),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.25, 1),
        "max_delta_step": trial.suggest_uniform("max_delta_step", 0, 10),
        
    }
    
    n_estimators = trial.suggest_int('n_estimators', 50, 100)

    # if param["booster"] == "gbtree" or param["booster"] == "dart":
    #     param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
    #     param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
    #     param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
    #     param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    # if param["booster"] == "dart":
    #     param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
    #     param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
    #     param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
    #     param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")],num_boost_round = n_estimators, callbacks=[pruning_callback])
    preds = bst.predict(dvalid)
    ERROR = np.sqrt(mean_squared_error(valid_y, preds))
    return ERROR



study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize")
study.optimize(objective, n_trials=100, n_jobs=CPUCOUNT)
print(study.best_trial)
pickle.dump(study,open(bestParFOLDER+'/STUDY.p','wb'))



###################################### TTTTTTTTEEEESSSSSSTTTTT
OUTPUTFOLDER = OuterFolder


X_train = X[train_index]
Y_train = Y[train_index]

X_test = X[test_index]
Y_test = Y[test_index]


params = study.best_trial.params
n_estimators = study.best_trial.params['n_estimators']
del params['n_estimators']
params['seed'] = 99

X_train_t, X_train_v, Y_train_t, Y_train_v = train_test_split(X_train, Y_train,test_size = 0.1, random_state = 1)

d_train = xgb.DMatrix(X_train_t,Y_train_t)
d_validate = xgb.DMatrix(X_train_v,Y_train_v)
d_test = xgb.DMatrix(X_test)

watchlist = [(d_train, 'train'), (d_validate, 'valid')]

model = xgb.train(params,d_train,num_boost_round = n_estimators,evals = watchlist)

# Accuracy on the test set
Y_predict = model.predict(d_test, ntree_limit=model.best_ntree_limit)
acc = accuracy_dilute(Y_predict,Y_test)
RMSE = np.sqrt(mean_squared_error(Y_predict,Y_test))
print(acc)



pickle.dump(Y_test, open(OUTPUTFOLDER+"/y_test-cv"+str(FOLDn)+".p", "wb"))
pickle.dump(Y_predict, open(OUTPUTFOLDER+"/Y_predict-cv"+str(FOLDn)+".p", "wb"))
pickle.dump(acc, open(OUTPUTFOLDER+"/acc-cv"+str(FOLDn)+".p", "wb"))
pickle.dump(RMSE, open(OUTPUTFOLDER+"/RMSE-cv"+str(FOLDn)+".p", "wb"))
pickle.dump(params, open(OUTPUTFOLDER+"/params-cv"+str(FOLDn)+".p", "wb"))
pickle.dump(model, open(OUTPUTFOLDER+"/model-cv"+str(FOLDn)+".p", "wb"))
f = open(OUTPUTFOLDER+"/ACCU.txt", "w")
f.write(str(acc)+"\n")
f.close()
