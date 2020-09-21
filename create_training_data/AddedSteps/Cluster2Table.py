#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:48:43 2020

@author: taha
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle


INPUTFOLDER = sys.argv[1]
OUTPUTFOLDER = sys.argv[2]


print('Reading the data...') 
INPUT = pd.read_csv(OUTPUTFOLDER+'/OUT_cluster.tsv',sep='\t',header=None)
 

InputFiles = os.listdir(INPUTFOLDER)

GENOMES = []

for file in InputFiles:
    if file.endswith('.faa'):
        GENOMES.append(file)
        
print('Done!')


print('Processing...') 
CLUSTERS = set(INPUT.iloc[:,0])
DATAPOINTS = set(INPUT.values.reshape(INPUT.shape[0]*INPUT.shape[1]))


ClustInd ={}
for ii,cluster in enumerate(CLUSTERS):
    ClustInd[cluster] = ii

GenomesInd ={}
for ii,genome in enumerate(GENOMES):
    GenomesInd[genome] = ii



TABLE = np.zeros([len(GENOMES),len(CLUSTERS)]).astype('int')

for ii in INPUT.index:
    
#    print(str(ii)+'/'+str(INPUT.shape[0]))
    
    cluster = INPUT.iloc[ii,0]
    genome = INPUT.iloc[ii,1].split('$')[0]
    
    TABLE[GenomesInd[genome],ClustInd[cluster]] = TABLE[GenomesInd[genome],ClustInd[cluster]] + 1

OUTPUT = pd.DataFrame(index=GENOMES, columns=CLUSTERS, data = TABLE)

print('Done!')

print('Saving output ...') 
pickle.dump(OUTPUT,open(OUTPUTFOLDER+'/GeneCount.p','wb'),protocol=4)
print('Done!') 
