#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 01:48:47 2020

@author: taha
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics


FOLDERs = ['/pylon5/br5phhp/tv349/AMR/OUTC',
           '/pylon5/br5phhp/tv349/AMR/OUTC',
           '/pylon5/br5phhp/tv349/AMR/OUTC',
           '/pylon5/br5phhp/tv349/AMR/OUTC',
           '/pylon5/br5phhp/tv349/AMR/OUTP',
           '/pylon5/br5phhp/tv349/AMR/OUTP',
           '/pylon5/br5phhp/tv349/AMR/OUTP',
           '/pylon5/br5phhp/tv349/AMR/OUTG',
           '/pylon5/br5phhp/tv349/AMR/OUTSNP',
           '/pylon5/br5phhp/tv349/AMR/OUTGSNP']

NumberOfKmerData = 7

Klist = [8,9,10,11,3,4,5]

LEGEND = ['NT 8-mers','NT 9-mers','NT 10-mers','NT 11-mers'\
              ,'AA 3-mers', 'AA 4-mers','AA 5-mers',\
                  'Gene content','SNP','Gene+SNP']


SPE = 'CampylobacterJejuni'

ABLIST = ['erythromycin', 'azithromycin', 'gentamicin', 'clindamycin', 'telithromycin',
          'ciprofloxacin', 'nalidixicacid', 'tetracycline', 'florfenicol']




f = open('/pylon5/br5phhp/tv349/AMR/Breakpoints/'+SPE+'.txt','r')
BRdic = f.read()
f.close()
PCommand = 'BreakPoints = {'+BRdic+'}'
exec(PCommand)


Kfold = 10

ANTIB = []

for ii in range(len(ABLIST)):
    ANTIB.append(SPE+'-'+ABLIST[ii])


DF = pd.DataFrame(index=LEGEND,columns=['RMSE','DD1','DD2','ME','VME'])

# calculate the accuracy based on the ±2 dilution
def accuracy_dilute(prdc,target):
    total = len(prdc);
    n = 0
    for i,j in zip(prdc,target):
        if j-1 <= i and i <= j+1:
            n += 1
    return n/total

def accuracy_dilute2(prdc,target):
    total = len(prdc);
    n = 0
    for i,j in zip(prdc,target):
        if j-2 <= i and i <= j+2:
            n += 1
    return n/total

for ii in ANTIB:

    print(ii)

    DF = pd.DataFrame(index=LEGEND,columns=['RMSE-CV','RMSE-H','DD1-CV','DD1-H','DD2-CV','DD2-H','ME-CV','ME-H','VME-CV','VME-H'])


    for COUNTER in range(len(FOLDERs)):


        cwd = FOLDERs[COUNTER]


        
        CleanName = []
        DATA1 = []
        HO1 = []




        
        
        VME = []
        ME = []
        RMSE = []
        DD1 = []
        DD2 = []
        
        for fold in range(Kfold):
    
            if COUNTER < NumberOfKmerData:
                k = Klist[COUNTER]
                try:
                    PRED = pickle.load(open(cwd+'/'+ii+'/k'+str(k)+'/Y_predict-cv'+str(fold)+'.p','rb'))
                    TEST = pickle.load(open(cwd+'/'+ii+'/k'+str(k)+'/y_test-cv'+str(fold)+'.p','rb'))

                    PREDho = pickle.load(open(cwd+'/'+ii+'/k'+str(k)+'/Held-Out-Eval/Y_predict.p','rb'))
                    TESTho = pickle.load(open(cwd+'/'+ii+'/k'+str(k)+'/Held-Out-Eval/y_test.p','rb'))

                except:
                    print(ii+"Doesn't exist")
                    continue
            else:
                try:
                    PRED = pickle.load(open(cwd+'/'+ii+'/Y_predict-cv'+str(fold)+'.p','rb'))
                    TEST = pickle.load(open(cwd+'/'+ii+'/y_test-cv'+str(fold)+'.p','rb'))

                    PREDho = pickle.load(open(cwd+'/'+ii+'/Held-Out-Eval/Y_predict.p','rb'))
                    TESTho = pickle.load(open(cwd+'/'+ii+'/Held-Out-Eval/y_test.p','rb'))

                    ACC = pd.read_csv(cwd+'/'+ii+'/ACCU.txt',header = None)
                    HO = pd.read_csv(cwd+'/'+ii+'/Held-Out-Eval/ACCU.txt',header = None)
                except:
                    print(ii+"Doesn't exist")
                    continue
            
            Low,Hi = np.log2(BreakPoints[ii.split('-')[1]])
            
            VMEfold = np.sum(np.logical_and(TEST>=Hi, PRED<=Low))/len(TEST)
            MEfold = np.sum(np.logical_and(PRED>=Hi, TEST<=Low))/len(TEST)
            
            RMSEfold = np.sqrt(metrics.mean_squared_error(PRED,TEST))
            
            DD1fold = accuracy_dilute(PRED,TEST)
            DD2fold = accuracy_dilute2(PRED,TEST)
            
            
            VME.append(VMEfold)
            ME.append(MEfold)
            RMSE.append(RMSEfold)
            DD1.append(DD1fold)
            DD2.append(DD2fold)

        
            VMEho = np.sum(np.logical_and(TESTho>=Hi, PREDho<=Low))/len(TESTho)
            MEho = np.sum(np.logical_and(PREDho>=Hi, TESTho<=Low))/len(TESTho)

            RMSEho = np.sqrt(metrics.mean_squared_error(PREDho,TESTho))

            DD1ho = accuracy_dilute(PREDho,TESTho)
            DD2ho = accuracy_dilute2(PREDho,TESTho)


        DF.loc[LEGEND[COUNTER],'RMSE-CV'] = str(np.round(np.mean(RMSE),3))+' ('+str(np.round(np.std(RMSE),3))+')'
        DF.loc[LEGEND[COUNTER],'DD1-CV'] = str(np.round(np.mean(DD1),3))+' ('+str(np.round(np.std(DD1),3))+')'
        DF.loc[LEGEND[COUNTER],'DD2-CV'] = str(np.round(np.mean(DD2),3))+' ('+str(np.round(np.std(DD2),3))+')'
        DF.loc[LEGEND[COUNTER],'ME-CV'] = str(np.round(np.mean(ME),3))+' ('+str(np.round(np.std(ME),3))+')'
        DF.loc[LEGEND[COUNTER],'VME-CV'] = str(np.round(np.mean(VME),3))+' ('+str(np.round(np.std(VME),3))+')'

        DF.loc[LEGEND[COUNTER],'RMSE-H'] = np.round(RMSEho,3)
        DF.loc[LEGEND[COUNTER],'DD1-H'] = np.round(DD1ho,3)
        DF.loc[LEGEND[COUNTER],'DD2-H'] = np.round(DD2ho,3)
        DF.loc[LEGEND[COUNTER],'ME-H']  = np.round(MEho,3)
        DF.loc[LEGEND[COUNTER],'VME-H'] = np.round(VMEho,3) 

        
    DF.to_csv('/pylon5/br5phhp/tv349/AMR/CompareBox/Error/Error-'+ii+'.tsv',sep='\t')
