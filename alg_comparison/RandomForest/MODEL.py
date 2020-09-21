#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:19:11 2020

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
from sklearn.ensemble import RandomForestRegressor
import os
import sys
import csv
from collections import defaultdict
import pickle
import gc
from sklearn import metrics
import shap
import random
from scipy.sparse import csr_matrix
print('Packages imported!') 


INPUTFOLDER = sys.argv[1]
OUTPUTFOLDER = sys.argv[2]
ParID = sys.argv[3]
NCPU = sys.argv[4]
FullName = sys.argv[5]
Sparse = sys.argv[6]

# Recursively create the output folder
def CreateFolder(FOLDER):
    
    BRK = FOLDER.split('/')
    
    ADD = ''
    for ii in range(len(BRK)):
        ADD = ADD + BRK[ii] + '/'
        os.system('mkdir '+ADD)
    return ADD

ADDR = CreateFolder(OUTPUTFOLDER)

#Create Heldout Folder
HeldOutAdd = ADDR+'HeldOut'
os.system('mkdir '+HeldOutAdd)


# calculate the accuracy based on the ±2 dilution
def accuracy_dilute(prdc,target):
    total = len(prdc);
    n = 0
    for i,j in zip(prdc,target):
        if j-1 <= i and i <= j+1:
            n += 1  
    return n/total


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
#Fix MIC feature issue
if np.sum(df.columns=='MIC')>1:
    LogicalMICpos = df.columns=='MIC'
    from itertools import compress
    MICpos = list(compress(range(len(LogicalMICpos)), LogicalMICpos))
    if len(MICpos)>1:
        WronPos = MICpos[-1]
        FixedColumn = pd.DataFrame(index = df.index,columns = ['MICf'])
        FixedColumn['MICf'] = df.iloc[:,WronPos]
        df = pd.concat([df.iloc[:,:WronPos], FixedColumn, df.iloc[:,WronPos+1:]], axis=1)



# All indices:
INDonMETA = np.arange(df.shape[0])


# training

RAWdfFEATURES = df.columns.values[2:]
pickle.dump(RAWdfFEATURES , open(OUTPUTFOLDER+'/Features.p','wb') )


print('Seperate a held-out set...')

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

INDonMETA = INDonMETA[KeepMask]


# ID of the held-out set
SeenID = pickle.load(open('/pylon5/br5phhp/tv349/AMR/BESTPAR/'+ParID+'/HeldOut/Train_ID.p', "rb"))
N2I = pickle.load(open("/pylon5/br5phhp/tv349/AMR/IndNameMap/"+FullName+"-N2I.p", "rb"))
# Remove Seen samples that do not exist in this antibiotic
SeenIDInThisAB = []
for ii in SeenID:
    if ii in N2I.keys():
        SeenIDInThisAB.append(ii)
SeenInd = [N2I[id] for id in SeenIDInThisAB]

Candidates = list(np.delete(INDonMETA,SeenInd))


HOratio = 0.1
RequiredHOSize = int(np.floor(len(INDonMETA)*HOratio))
if RequiredHOSize >= len(Candidates):
    Test_ind = Candidates
else:
    Test_ind = random.sample(Candidates,RequiredHOSize)

Train_ind = list(np.delete(INDonMETA,Test_ind))


X = Xfull[Train_ind]
Y = Yfull[Train_ind]

pickle.dump(Train_ind, open(HeldOutAdd+"/Train_ind.p", "wb"))
pickle.dump(Test_ind, open(HeldOutAdd+"/Test_ind.p", "wb"))

print('Held-out set saved')


##Stratified croess-validation
acc_list = [];

#set up stratified k-fold
kfold = 10
skf = StratifiedKFold(n_splits=kfold,random_state=False,shuffle=False) 
# skf = KFold(n_splits=kfold)

#params = pickle.load(open('/pylon5/br5phhp/tv349/AMR/BESTPAR/'+ParID+'/LeastErrorPar.p', "rb"))
#n_estimators = int(params['n_estimators'])
#del params['n_estimators']
#params['nthread'] = NCPU
#params['seed'] = 99
#params['objective'] = 'reg:squarederror'


CLASS = np.zeros(Y.shape)
for ii,element in enumerate(set(Y)):
    CLASS[Y==element] = ii


# Create ACC file
f = open(OUTPUTFOLDER+"/ACCU.txt", "w")
f.close()


# Create ACC train file
f = open(OUTPUTFOLDER+"/ACCUtrain.txt", "w")
f.close()


#10-fold cross validation
for i, (train_index, test_index) in enumerate(skf.split(X, CLASS)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    CLASS_train, CLASS_test = CLASS[train_index], CLASS[test_index]
    

    X_train_t, X_train_v, Y_train_t, Y_train_v = train_test_split(X_train, Y_train,test_size = 0.1, random_state = 1)

    if Sparse:
        d_train = xgb.DMatrix(csr_matrix(X_train_t),Y_train_t)
        d_validate = xgb.DMatrix(csr_matrix(X_train_v),Y_train_v)
        d_test = xgb.DMatrix(csr_matrix(X_test))
    else:    
        d_train = xgb.DMatrix(X_train_t,Y_train_t)
        d_validate = xgb.DMatrix(X_train_v,Y_train_v)
        d_test = xgb.DMatrix(X_test)
    
    #watchlist = [(d_train, 'train'), (d_validate, 'valid')]
    
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train,Y_train)
    pickle.dump(model, open(OUTPUTFOLDER+"/GXMODEL-cv"+str(i)+".dat", "wb"))
    
    # Accuracy on the test set
    Y_predict = model.predict(X_test)
    acc = accuracy_dilute(Y_predict,Y_test)
    print(acc)
    acc_list.append(acc)
    
    # Accuracy on the train set
    #d_train_test = xgb.DMatrix(X_train_t);
    Train_predict = model.predict(X_train);
    Trainacc = accuracy_dilute(Train_predict,Y_train);
    
    
    plt.figure(i)
    plt.plot(Y_test,Y_predict,'.')
    
    x = np.array(range(int(np.min(Y_test)),1+int(np.max(Y_test))))
    plt.plot(x,x,color='g')
    plt.plot(x,x-1,color='r')
    plt.plot(x,x+1,color='r')
    
    plt.xlabel('Actal value (log2 scale)')
    plt.ylabel('Predicted value (log2 scale)')
    plt.title('Predicted values versus actual values. Accuracy:'+str(acc))
    plt.legend(['Data','Perfect prediction','Two-fold dilution'])
    
    plt.savefig(OUTPUTFOLDER+"/ACC"+str(i)+".jpg",format='jpg')
    
    #explainer = shap.TreeExplainer(model)
    #shap_values = explainer.shap_values(X_train_t)
    #plt.figure(i+kfold)
    #shap.summary_plot(shap_values, X_train_t)
    #plt.tight_layout()
    #plt.savefig(OUTPUTFOLDER+"/SHAP"+str(i)+".jpg",format='jpg')
    #FeatureOrder = np.flip(np.argsort(np.sum(np.abs(shap_values),axis=0)))
    #FeatureShapMag = np.sum(np.abs(shap_values),axis=0)
    #NonZeroSHAPFeat = np.where(np.sum(np.abs(shap_values),axis=0)!=0)[0]
    #ShapDataCor = dict([])
    #for rr in NonZeroSHAPFeat:
    #    ShapDataCor[rr] = np.corrcoef(shap_values[:,rr],X_train_t[:,rr])[0,1]
        
    pickle.dump(train_index, open(OUTPUTFOLDER+"/train_index-cv"+str(i)+".p", "wb"))
    pickle.dump(test_index, open(OUTPUTFOLDER+"/test_index-cv"+str(i)+".p", "wb"))
    #pickle.dump(FeatureShapMag, open(OUTPUTFOLDER+"/FeatureShapMag-cv"+str(i)+".p", "wb"))
    #pickle.dump(ShapDataCor, open(OUTPUTFOLDER+"/ShapDataCor-cv"+str(i)+".p", "wb"))
    pickle.dump(Y_test, open(OUTPUTFOLDER+"/y_test-cv"+str(i)+".p", "wb"))
    pickle.dump(Y_predict, open(OUTPUTFOLDER+"/Y_predict-cv"+str(i)+".p", "wb"))
    #pickle.dump(FeatureOrder, open(OUTPUTFOLDER+"/FeatureOrder-cv"+str(i)+".p", "wb"))
    #try:
    #    np.save(OUTPUTFOLDER+"/shap_values-cv"+str(i), shap_values)
    #    
    #except:
    #    print('Big File numpy save error!')

    f1 = open(OUTPUTFOLDER+"/ACCUtrain.txt", "a")
    f1.write(str(Trainacc)+"\n")
    f1.close()
 	
    f = open(OUTPUTFOLDER+"/ACCU.txt", "a")
    f.write(str(acc)+"\n")
    f.close()


print(acc_list)
print(np.mean(acc_list))



#plot

PrintClassAcc = True

OutputDir = OUTPUTFOLDER
cwd = OUTPUTFOLDER
KFOLD = kfold

Y_predict = []
Y_test = []
for CVcount in range(KFOLD):
    
    Y_predictFOLD = pickle.load(open(cwd+'/Y_predict-cv'+str(CVcount)+'.p', "rb"))
    Y_testFOLD = pickle.load(open(cwd+'/y_test-cv'+str(CVcount)+'.p', "rb"))
    
    Y_predict = np.concatenate([Y_predict,Y_predictFOLD])
    Y_test = np.concatenate([Y_test,Y_testFOLD])

YList = sorted(list(set(Y_test)))

#    CONF = np.zeros([len(YList),len(YList)])
#    for ii in range(len(Y_predict)):
#        CONF[YList.index(Y_predict[ii]),YList.index(Y_test[ii])] += 1
#        
#    X = []
#    Y = []
#    R = []
#    for ii in YList:
#        for jj in YList:
#            X.append(ii)
#            Y.append(jj)
#            R.append(CONF[YList.index(ii),YList.index(jj)])
    
#    plt.figure()
#    plt.scatter(X,Y,s=R)

x = np.array(range(int(np.min(Y_test)),1+int(np.max(Y_test))))
plt.plot(x,x,color='g')

Z1 = np.polyfit(Y_test,Y_predict,1)
P1 = np.poly1d(Z1)
R2 = metrics.r2_score(Y_predict,P1(Y_test))
plt.plot(Y_test,P1(Y_test),color='y')


plt.plot(x,x-1,color='r')
plt.plot(x,x+1,color='r')

plt.xlabel('Actual value (log2 scale)')
plt.ylabel('Predicted value (log2 scale)')
plt.title('. Predicted vs actual values. Accuracy:'+\
          str(np.round(accuracy_dilute(Y_predict,Y_test),4)))
plt.legend(['Perfect prediction','First order regression (R^2 score:'+str(np.round(R2,2))+')','Two-fold dilution'])

plt.savefig(OutputDir+"/Scatter.jpg",format='jpg')
plt.savefig(OutputDir+"/Scatter.pdf",format='pdf')


ActualValues = list(set(Y_test))
ActualValues.sort()
NumberOfValues = len(ActualValues)
MICDIC = {}
for VAL in ActualValues:
    MICDIC[VAL] = ActualValues.index(VAL)


VIOLIN = plt.figure()
DATA = [[] for _ in range(NumberOfValues)]
ACTUALDATA = [[] for _ in range(NumberOfValues)]

for ii in range(len(Y_test)):
    predic = Y_predict[ii]
    test = Y_test[ii]
    
    DATA[MICDIC[test]].append(predic)
    ACTUALDATA[MICDIC[test]].append(test)
    
ACCclass = []
NumberDataClass = []
for ii in range(NumberOfValues):
    ACCclass.append(accuracy_dilute(DATA[ii],ACTUALDATA[ii]))
    NumberDataClass.append(len(DATA[ii]))

plt.violinplot(DATA,showmeans=True,showextrema=True,positions=ActualValues)
plt.plot(x,x,color='g')

Z1 = np.polyfit(Y_test,Y_predict,1)
P1 = np.poly1d(Z1)
R2 = metrics.r2_score(Y_predict,P1(Y_test))
plt.plot(Y_test,P1(Y_test),color='y')

plt.plot(x,x-1,color='r')
plt.plot(x,x+1,color='r')
plt.xlabel('Actual value (log2 scale)')
plt.ylabel('Predicted value (log2 scale)')
plt.title('. Predicted vs actual values. Accuracy:'+\
          str(np.round(accuracy_dilute(Y_predict,Y_test),4)))
plt.legend(['Perfect prediction','First order regression (R^2 score:'+str(np.round(R2,2))+')','Two-fold dilution'])

if PrintClassAcc:
        
    for ii in range(len(ACCclass)):
        STR = str(np.round(ACCclass[ii],2))
        STR2 = '('+str(NumberDataClass[ii])+')'
        plt.text(ActualValues[ii]-0.2,np.min(np.min(DATA))-1,STR)
        plt.text(ActualValues[ii]-0.2,np.min(np.min(DATA))-1.3,STR2)


plt.savefig(OutputDir+"/Violin.jpg",format='jpg')
plt.savefig(OutputDir+"/Violin.pdf",format='pdf')
pickle.dump(VIOLIN, open(OutputDir+"/Violin.pickle", "wb"))


##############################################################################
#Held-Out Evaluation:
OUTPUTFOLDER = OUTPUTFOLDER+'/Held-Out-Eval'
os.system('mkdir '+OUTPUTFOLDER)


print('Hel-Out evaluation')


# training
    
RAWdfFEATURES = df.columns.values[2:]
pickle.dump(RAWdfFEATURES , open(OUTPUTFOLDER+'/Features.p','wb') )




Train_ind = pickle.load(open(HeldOutAdd+"/Train_ind.p", "rb"))
Test_ind = pickle.load(open(HeldOutAdd+"/Test_ind.p", "rb"))

X_train, X_test = Xfull[Train_ind], Xfull[Test_ind]
Y_train, Y_test = Yfull[Train_ind], Yfull[Test_ind]

X_train_t, X_train_v, Y_train_t, Y_train_v = train_test_split(X_train, Y_train,test_size = 0.1, random_state = 1)


if Sparse:
    d_train = xgb.DMatrix(csr_matrix(X_train_t),Y_train_t)
    d_validate = xgb.DMatrix(csr_matrix(X_train_v),Y_train_v)
    d_test = xgb.DMatrix(csr_matrix(X_test))
else:
    d_train = xgb.DMatrix(X_train_t,Y_train_t)
    d_validate = xgb.DMatrix(X_train_v,Y_train_v)
    d_test = xgb.DMatrix(X_test)

#watchlist = [(d_train, 'train'), (d_validate, 'valid')]

model = RandomForestRegressor(random_state=0)
model.fit(X_train,Y_train)
pickle.dump(model, open(OUTPUTFOLDER+"/GXMODEL.dat", "wb"))

# Accuracy on the test set
Y_predict = model.predict(X_test)
acc = accuracy_dilute(Y_predict,Y_test)
print(acc)
FINALACC = acc

# Accuracy on the train set
#d_train_test = xgb.DMatrix(X_train_t);
Train_predict = model.predict(X_train);
Trainacc = accuracy_dilute(Train_predict,Y_train);


plt.figure()
plt.plot(Y_test,Y_predict,'.')

x = np.array(range(int(np.min(Y_test)),1+int(np.max(Y_test))))
plt.plot(x,x,color='g')
plt.plot(x,x-1,color='r')
plt.plot(x,x+1,color='r')

plt.xlabel('Actal value (log2 scale)')
plt.ylabel('Predicted value (log2 scale)')
plt.title('Predicted values versus actual values. Accuracy:'+str(acc))
plt.legend(['Data','Perfect prediction','Two-fold dilution'])

plt.savefig(OUTPUTFOLDER+"/ACC.jpg",format='jpg')

#explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(X)
#plt.figure()
#shap.summary_plot(shap_values, X)
#plt.tight_layout()
#plt.savefig(OUTPUTFOLDER+"/SHAP.jpg",format='jpg')
#FeatureOrder = np.flip(np.argsort(np.sum(np.abs(shap_values),axis=0)))
#FeatureShapMag = np.sum(np.abs(shap_values),axis=0)
#ShapDataCor = np.empty(shap_values.shape[1])
#for rr in range(shap_values.shape[1]):
#    ShapDataCor[rr] = np.corrcoef(shap_values[:,rr],X[:,rr])[0,1]
    
pickle.dump(Train_ind, open(OUTPUTFOLDER+"/train_index.p", "wb"))
pickle.dump(Test_ind, open(OUTPUTFOLDER+"/test_index.p", "wb"))
pickle.dump(Y_test, open(OUTPUTFOLDER+"/y_test.p", "wb"))
pickle.dump(Y_predict, open(OUTPUTFOLDER+"/Y_predict.p", "wb"))
#pickle.dump(FeatureShapMag, open(OUTPUTFOLDER+"/FeatureShapMag.p", "wb"))
#pickle.dump(ShapDataCor, open(OUTPUTFOLDER+"/ShapDataCor.p", "wb"))
#pickle.dump(FeatureOrder, open(OUTPUTFOLDER+"/FeatureOrder.p", "wb"))

f1 = open(OUTPUTFOLDER+"/ACCUtrain.txt", "w")
f1.write(str(Trainacc)+"\n")
f1.close()
 	
f = open(OUTPUTFOLDER+"/ACCU.txt", "w")
f.write(str(acc)+"\n")
f.close()







#plot

PrintClassAcc = True

OutputDir = OUTPUTFOLDER
cwd = OUTPUTFOLDER
KFOLD = kfold


    
Y_predict =  pickle.load(open(cwd+'/Y_predict.p', "rb"))
Y_test = pickle.load(open(cwd+'/y_test.p', "rb"))

YList = sorted(list(set(Y_test)))

#    CONF = np.zeros([len(YList),len(YList)])
#    for ii in range(len(Y_predict)):
#        CONF[YList.index(Y_predict[ii]),YList.index(Y_test[ii])] += 1
#        
#    X = []
#    Y = []
#    R = []
#    for ii in YList:
#        for jj in YList:
#            X.append(ii)
#            Y.append(jj)
#            R.append(CONF[YList.index(ii),YList.index(jj)])
    
#    plt.figure()
#    plt.scatter(X,Y,s=R)

x = np.array(range(int(np.min(Y_test)),1+int(np.max(Y_test))))
plt.plot(x,x,color='g')

Z1 = np.polyfit(Y_test,Y_predict,1)
P1 = np.poly1d(Z1)
R2 = metrics.r2_score(Y_predict,P1(Y_test))
plt.plot(Y_test,P1(Y_test),color='y')


plt.plot(x,x-1,color='r')
plt.plot(x,x+1,color='r')

plt.xlabel('Actual value (log2 scale)')
plt.ylabel('Predicted value (log2 scale)')
plt.title('. Predicted vs actual values. Accuracy:'+\
          str(np.round(accuracy_dilute(Y_predict,Y_test),4)))
plt.legend(['Perfect prediction','First order regression (R^2 score:'+str(np.round(R2,2))+')','Two-fold dilution'])

plt.savefig(OutputDir+"/Scatter.jpg",format='jpg')
plt.savefig(OutputDir+"/Scatter.pdf",format='pdf')


ActualValues = list(set(Y_test))
ActualValues.sort()
NumberOfValues = len(ActualValues)
MICDIC = {}
for VAL in ActualValues:
    MICDIC[VAL] = ActualValues.index(VAL)


VIOLIN = plt.figure()
DATA = [[] for _ in range(NumberOfValues)]
ACTUALDATA = [[] for _ in range(NumberOfValues)]

for ii in range(len(Y_test)):
    predic = Y_predict[ii]
    test = Y_test[ii]
    
    DATA[MICDIC[test]].append(predic)
    ACTUALDATA[MICDIC[test]].append(test)
    
ACCclass = []
NumberDataClass = []
for ii in range(NumberOfValues):
    ACCclass.append(accuracy_dilute(DATA[ii],ACTUALDATA[ii]))
    NumberDataClass.append(len(DATA[ii]))

plt.violinplot(DATA,showmeans=True,showextrema=True,positions=ActualValues)
plt.plot(x,x,color='g')

Z1 = np.polyfit(Y_test,Y_predict,1)
P1 = np.poly1d(Z1)
R2 = metrics.r2_score(Y_predict,P1(Y_test))
plt.plot(Y_test,P1(Y_test),color='y')

plt.plot(x,x-1,color='r')
plt.plot(x,x+1,color='r')
plt.xlabel('Actual value (log2 scale)')
plt.ylabel('Predicted value (log2 scale)')
plt.title('. Predicted vs actual values. Accuracy:'+\
          str(np.round(accuracy_dilute(Y_predict,Y_test),4)))
plt.legend(['Perfect prediction','First order regression (R^2 score:'+str(np.round(R2,2))+')','Two-fold dilution'])

if PrintClassAcc:
        
    for ii in range(len(ACCclass)):
        STR = str(np.round(ACCclass[ii],2))
        STR2 = '('+str(NumberDataClass[ii])+')'
        plt.text(ActualValues[ii]-0.2,np.min(np.min(DATA))-1,STR)
        plt.text(ActualValues[ii]-0.2,np.min(np.min(DATA))-1.3,STR2)


plt.savefig(OutputDir+"/Violin.jpg",format='jpg')
plt.savefig(OutputDir+"/Violin.pdf",format='pdf')
pickle.dump(VIOLIN, open(OutputDir+"/Violin.pickle", "wb"))
