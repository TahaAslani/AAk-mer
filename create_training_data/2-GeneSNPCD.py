#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:13:16 2019

@author: taha
"""

import numpy as np
import pandas as pd
import datetime
import os
from xlrd import open_workbook
import pickle
import sys


Species = sys.argv[1]
NPrc = int(sys.argv[2])
Thread = int(sys.argv[3])


print('GENE COUNT FOR '+Species.upper())

MetaFileName = Species+'.xlsx'


try:
    os.system('mkdir /lustre/scratch/tv349/AMR/GeneTrainingData/'+Species)
except:
    print('Destenation Folder exists')



print('Loading Metadata...')
META = pd.read_excel('/lustre/scratch/tv349/AMR/'+MetaFileName)
wb = open_workbook('/lustre/scratch/tv349/AMR/'+MetaFileName)
for sheet in wb.sheets():
    t = 6
print('Metadata file loaded')

print('Data pre-processing...')

# Correct names
for counter in range(META.shape[0]):
    if counter % int((META.shape[0])/50) == 0:
        print('#')
    META.iloc[counter,1] = sheet.cell(1+counter,1).value
#    print(META.iloc[counter,1])

#META = META.loc[~META.iloc[:,7].isnull() & (META.iloc[:,9] == 'MIC'),:]
print('Done!')


GeneCount = pickle.load( open( "/lustre/scratch/tv349/AMR/MMSEQ2/"+\
                                   Species.split('-')[0]+'/GeneCount.p', "rb" ) )

KEYS = []
for FEAT in GeneCount.columns:
    FeatName = 'G-'+str(FEAT)
    KEYS.append(FeatName)


T2=datetime.datetime.now()

COL = ['Antibiotic','MIC']

Q = int(np.ceil(META.shape[0] / NPrc))


    
print('Thread# : '+str(Thread))

print('Creating empty array... ')

ChunkLen = len(range(Q*Thread,min(Q*(Thread+1),META.shape[0])))

TABLEmeta = pd.DataFrame(np.zeros([ChunkLen,len(COL)]), columns=COL)
#filename = path.join(mkdtemp(), '/lustre/scratch/tv349/AMR/Kmer/k'+str(K)+'/DISKTABLE.dat')
#TABLE = ap(filename, dtype='float32', mode='w+',\
#                  shape=(ChunkLen,len(KEYS)))
TABLE = np.zeros([ChunkLen,len(KEYS)])
print('Done')



for counter in range(Q*Thread,min(Q*(Thread+1),META.shape[0])):
    
    LocalCounter = counter % Q

    # Check fo NaN
    if str(META.iloc[counter,7]) == 'nan':
        continue
    
    print(str(datetime.datetime.today().replace(microsecond=0))+\
          ' ['+str(counter)+'/'+str(META.shape[0])+'] ')
    print('Thread #'+str(Thread)+'-ADDING DATA: '+\
          str(META.iloc[counter,1])+'. MIC: '+\
          str(META.iloc[counter,7]))
    
    READID = str(META.iloc[counter,1])
    Antibiotic = META.iloc[counter,3]
    MIC = META.iloc[counter,7]
    SIGN = META.iloc[counter,6]
    

    if SIGN == '<':
        MIC = str(float(MIC)/2)
    if SIGN == '>':
        MIC = str(float(MIC)*2)
    
    
    MIC = str(np.log2(float(MIC)))

    
    TABLEmeta.loc[LocalCounter,'Antibiotic'] = Antibiotic
    TABLEmeta.loc[LocalCounter,'MIC'] = MIC
    

    TABLE[LocalCounter,:] = GeneCount.loc[READID+'.PATRIC.faa']


# Add column names
TABLEdfGENE = pd.DataFrame(TABLE , columns=KEYS)







SNP = pickle.load( open( "/lustre/scratch/tv349/AMR/SNP/"+\
                                   Species.split('-')[0]+'/SNPOneHot.p', "rb" ) )
KEYS = []
for FEAT in SNP.columns:
    FeatName = 'SNP-'+str(FEAT)
    KEYS.append(FeatName)

Q = int(np.ceil(META.shape[0] / NPrc))


print('Thread# : '+str(Thread))

print('Creating empty array... ')

ChunkLen = len(range(Q*Thread,min(Q*(Thread+1),META.shape[0])))

TABLEsnp = np.zeros([ChunkLen,len(KEYS)])
print('Done')



for counter in range(Q*Thread,min(Q*(Thread+1),META.shape[0])):

    LocalCounter = counter % Q

    # Check fo NaN
    if str(META.iloc[counter,7]) == 'nan':
        continue

    print(str(datetime.datetime.today().replace(microsecond=0))+\
          ' ['+str(counter)+'/'+str(META.shape[0])+'] ')
    print('Thread #'+str(Thread)+'-ADDING DATA: '+\
          str(META.iloc[counter,1])+'. MIC: '+\
          str(META.iloc[counter,7]))
    
    READID = str(META.iloc[counter,1])
    
    
    TABLEsnp[LocalCounter,:] = SNP.loc[READID]

TABLEdfsnp = pd.DataFrame(TABLEsnp , columns=KEYS)


# Attach meta data
FINALTABLE = pd.concat([TABLEmeta, TABLEdfGENE, TABLEdfsnp], axis=1)


OutpuFilePickle = '/lustre/scratch/tv349/AMR/GeneSNPTrainingData/'+Species+\
'/GENESNPTRAININGDATA.'+MetaFileName+'-Part'+str(Thread)+'.p'
print('Saving data to file '+OutpuFilePickle)
with open( OutpuFilePickle , "wb" ) as pf:
    pickle.dump( FINALTABLE, pf, protocol=4)


OutpuFilePickle = '/lustre/scratch/tv349/AMR/GeneSNPTrainingData/'+Species+\
'/GENESNPTRAININGDATA.'+MetaFileName+'-Part'+str(Thread)+'.tsv'
print('Saving data to file '+OutpuFilePickle)
FINALTABLE.to_csv(OutpuFilePickle,sep='\t',index=False)
print('Done!')

print('Done!')

## Run parallel
#if __name__ == '__main__':
#    p = Pool(NPrc)
#    p.map(ProcessChunk, list(range(NPrc)))

T3=datetime.datetime.now()
print('All done!')
print('Table creatoin time: '+str(T3-T2))
