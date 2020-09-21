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
import pickle
import sys
import multiprocessing
from xlrd import open_workbook



Species = sys.argv[1]
K = int(sys.argv[2])
NPrc = int(sys.argv[3])
Thread = int(sys.argv[4])


print('KMC3 NT KMER COUNT FOR '+Species.upper()+', K = '+str(K))

MetaFileName = Species+'.xlsx'


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
print('pre-processing done')

print('Metadata file loaded')

os.chdir('/lustre/scratch/tv349/AMR/PROTEIN'+Species+str(K))

T2=datetime.datetime.now()


CPUcount = 4


Q = int(np.ceil(META.shape[0] / NPrc))


    
print('Thread# : '+str(Thread))



for counter in range(Q*Thread,min(Q*(Thread+1),META.shape[0])):
    
    LocalCounter = counter % Q

    READID = str(META.iloc[counter,1])
    
    print(str(datetime.datetime.today().replace(microsecond=0))+\
          ' ['+str(counter)+'/'+str(META.shape[0])+'] ')
    print('Thread #'+str(Thread)+'-ADDING DATA: '+\
          str(READID))
    
    COMMAND = "mercat -i /lustre/scratch/tv349/AMR/PROTEIN"+Species+str(K)+"/"+\
    str(READID)+".PATRIC.faa -k "+str(K)+" -n "+str(CPUcount)+" -c 1  -pro"

    ErrorNumber = os.system(COMMAND+'>/dev/null 2>&1')

    if ErrorNumber:
        print("ERROR IN MERCAT")
        print(COMMAND)
        os.system(COMMAND)
    #            raise ValueError('A very specific bad thing happened.')





## Run parallel
#if __name__ == '__main__':
#    p = Pool(NPrc)
#    p.map(ProcessChunk, list(range(NPrc)))

T3=datetime.datetime.now()
print('All done!')
print('K: '+str(K))
print('Table creatoin time: '+str(T3-T2))
