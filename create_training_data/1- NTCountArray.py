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
CPUcount = int(sys.argv[5])


# Count non- canonical kmers (Otherwise reverse comilments won't be counted)
CountNonCanonical = True
if CountNonCanonical:
    REV = "-b"
else:
     REV = " "     

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

os.chdir('/lustre/scratch/tv349/AMR/Kmer/k'+str(K))

T2=datetime.datetime.now()




GENOMESset = set(META.iloc[:,1])
GENOMESunsorted = list(GENOMESset)
GENOMES = sorted(GENOMESunsorted)

Q = int(np.ceil(len(GENOMES) / NPrc))

print(GENOMES)
    
print('Thread# : '+str(Thread))

BadCommand = []

for counter in range(Q*Thread,min(Q*(Thread+1),len(GENOMES))):
    
    LocalCounter = counter % Q

    READID = str(GENOMES[counter])
    
    print(str(datetime.datetime.today().replace(microsecond=0))+\
          ' ['+str(counter)+'/'+str(len(GENOMES))+'] ')
    print('Thread #'+str(Thread)+'-ADDING DATA: '+\
          str(READID))
    
    COMMANDkmercount = "kmc -t"+str(CPUcount)+" -k"+str(K)+" -cs4294967295 -ci0 "+REV+" -fm "+\
    "/lustre/scratch/tv349/AMR/NT"+Species+"/"+str(READID)+".fna "+str(READID)+\
    ' /lustre/scratch/tv349/AMR/Kmer/k'+str(K)
    COMMAND2text = "kmc_tools transform "+str(READID)+" dump out"+str(READID)
    
#    ErrorNumber1 = os.system(COMMANDkmercount+'>/dev/null 2>&1')
#    ErrorNumber2 = os.system(COMMAND2text+'>/dev/null 2>&1')
    ErrorNumber1 = os.system(COMMANDkmercount)
    ErrorNumber2 = os.system(COMMAND2text)
    os.system(COMMANDkmercount)
    os.system(COMMAND2text)

    print(ErrorNumber1)
    print(ErrorNumber2)
    if ErrorNumber1 or ErrorNumber2:
        BadCommand.append(COMMANDkmercount)
        print(COMMANDkmercount)
        print(COMMAND2text)
        raise ValueError('A very specific bad thing happened.')





## Run parallel
#if __name__ == '__main__':
#    p = Pool(NPrc)
#    p.map(ProcessChunk, list(range(NPrc)))

T3=datetime.datetime.now()
print('All done!')
print('K: '+str(K))
print('Table creatoin time: '+str(T3-T2))
