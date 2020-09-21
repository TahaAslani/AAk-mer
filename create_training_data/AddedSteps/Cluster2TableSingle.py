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
from scipy.sparse import csr_matrix


OUTPUTFOLDER = sys.argv[1]



print('Reading the data...') 
INPUT = pd.read_csv(OUTPUTFOLDER+'/Clustering_cluster.tsv',sep='\t',header=None)
 


print('Processing...') 
CLUSTERS = set(INPUT.iloc[:,0])
DATAPOINTS = set(INPUT.values.reshape(INPUT.shape[0]*INPUT.shape[1]))


ClustInd ={}
for ii,cluster in enumerate(CLUSTERS):
    ClustInd[cluster] = ii

DatapointInd ={}
for ii,Datapoint in enumerate(DATAPOINTS):
    ClustInd[Datapoint] = ii

#TABLE = np.zeros([len(CLUSTERS),len(DATAPOINTS)]).astype('int')
TABLE = csr_matrix((len(CLUSTERS), len(DATAPOINTS)), dtype=np.bool).toarray()


for ii in INPUT.index:
    
#    print(str(ii)+'/'+str(INPUT.shape[0]))
    
    cluster = INPUT.iloc[ii,0]
    Datapoint = INPUT.iloc[ii,1]
    
    TABLE[ClustInd[cluster],DatapointInd[Datapoint]] = 1

OUTPUT = pd.DataFrame(index=CLUSTERS, columns=DATAPOINTS, data = TABLE)

print('Done!')

print('Saving output ...') 
pickle.dump(OUTPUT,open(OUTPUTFOLDER+'/GeneCount.p','wb'),protocol=4)
print('Done!') 
