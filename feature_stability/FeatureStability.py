#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 00:29:49 2020

@author: taha
"""

import pickle
import os
from scipy.stats import spearmanr
import numpy as np

    
folder = '/home/taha/Desktop/AMR/Feature_Stability/'

def SR(folder):
    FEATs = [[],[],[],[],[],[],[],[],[],[]]
    
    for ii in range(10):
        
        file = 'FeatureOrder-cv'+str(ii)+'.p'
        
        path = os.path.join(folder,file)
        FeatOrder = pickle.load(open(path,'rb'))
        
        FEATs[ii] = FeatOrder
    
    S = []
    for ii in range(10):
        for jj in range(ii+1,10):
            S.append(spearmanr(FEATs[ii],FEATs[jj])[0])
            
    Sr = np.mean(S)
    
    return Sr

def Tanimoto(S1,S2):
    
    size1 = len(S1)
    size2 = len(S2)
    
    size_intersection = len(S1.intersection(S2))
    
    return 1 - (size1 + size2 - 2*size_intersection)/(size1 + size2 - size_intersection)

def SS(folder):
    
    MaxFeat = 50
    
    FEATs = [[],[],[],[],[],[],[],[],[],[]]
    
    for ii in range(10):
        
        file = 'FeatureOrder-cv'+str(ii)+'.p'
        
        path = os.path.join(folder,file)
        FeatOrder = pickle.load(open(path,'rb'))
        
        FEATs[ii] = set(FeatOrder[:MaxFeat])
    
    S = []
    for ii in range(10):
        for jj in range(ii+1,10):
            S.append(Tanimoto(FEATs[ii],FEATs[jj]))
            
    Ss = np.mean(S)
    
    return Ss


P1 = '/pylon5/br5phhp/tv349/AMR/OUTG'

DATASETS = os.listdir(P1)

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
                  'Gene content','SNP','Gene content + SNP']


for COUNTER,folder in enumerate(FOLDERs):

    score = []
    for data in DATASETS:
    
        path = os.path.join(folder,data)

        if COUNTER < NumberOfKmerData:
            k = Klist[COUNTER]
            path = os.path.join(path,'k'+str(k))
        
        score.append(SS(path))
    
    print(LEGEND[COUNTER],np.mean(score))



