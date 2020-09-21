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



# Count non- canonical kmers (Otherwise reverse comilments won't be counted)
CountNonCanonical = True
if CountNonCanonical:
    REV = "-b"
else:
     REV = " "     

CPUcount = 1

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


KEYS = set()

for counter in range(len(GENOMES)):
    
    
    READID = str(GENOMES[counter])
    
    print(str(datetime.datetime.today().replace(microsecond=0))+\
          ' ['+str(counter)+'/'+str(len(GENOMES))+'] ')
    print('-ADDING DATA: '+\
          str(READID))
    
    try:
        KMCres = pd.read_csv('/lustre/scratch/tv349/AMR/Kmer/k'+str(K)+"/out"+str(READID),header=None,sep='\t')
        KEYS.update(list(KMCres.iloc[:,0]))
    except:
	COMMANDkmercount = "kmc -t"+str(CPUcount)+" -k"+str(K)+" -cs4294967295 -ci0 "+REV+" -fm "+\
        "/lustre/scratch/tv349/AMR/NT"+Species+"/"+str(READID)+".fna "+str(READID)+\
        ' /lustre/scratch/tv349/AMR/Kmer/k'+str(K)
        COMMAND2text = "kmc_tools transform "+str(READID)+" dump out"+str(READID)

        ErrorNumber1 = os.system(COMMANDkmercount+'>/dev/null 2>&1')
        ErrorNumber2 = os.system(COMMAND2text+'>/dev/null 2>&1')

        if ErrorNumber1 or ErrorNumber2:

            print(COMMANDkmercount)
            print(COMMAND2text)
            os.system(COMMANDkmercount)
            os.system(COMMAND2text)
            print(ErrorNumber1)
            print(ErrorNumber2)

        KMCres = pd.read_csv('/lustre/scratch/tv349/AMR/Kmer/k'+str(K)+"/out"+str(READID),header=None,sep='\t')
        KEYS.update(list(KMCres.iloc[:,0]))



ColNum = {}
ii = 0
for key in KEYS:
    ColNum[key] = ii
    ii = ii + 1


pickle.dump(KEYS,open('/lustre/scratch/tv349/AMR/KEYS/NT-'+Species+'-k'+str(K)+'KEYS.p','wb'))
pickle.dump(ColNum,open('/lustre/scratch/tv349/AMR/KEYS/NT-'+Species+'-k'+str(K)+'COLS.p','wb'))


## Run parallel
#if __name__ == '__main__':
#    p = Pool(NPrc)
#    p.map(ProcessChunk, list(range(NPrc)))

T3=datetime.datetime.now()
print('All done!')
print('K: '+str(K))
