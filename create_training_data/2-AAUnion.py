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

CPUcount = 1

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
        MERCATres = pd.read_csv('/lustre/scratch/tv349/AMR/PROTEIN'+Species+str(K)+'/mercat_results/'\
                                +str(READID)+".PATRIC_protein_run/"+str(READID)\
                                +".PATRIC_protein_summary.csv",header=0,sep=',')
    except:
        COMMAND = "mercat -i /lustre/scratch/tv349/AMR/PROTEIN"+Species+str(K)+"/"+\
        str(READID)+".PATRIC.faa -k "+str(K)+" -n "+str(CPUcount)+" -c 1  -pro"
    
        ErrorNumber = os.system(COMMAND+'>/dev/null 2>&1')
    
        if ErrorNumber:
            print("ERROR IN MERCAT")
            print(COMMAND)
            os.system(COMMAND)


        MERCATres = pd.read_csv('/lustre/scratch/tv349/AMR/PROTEIN'+Species+str(K)+'/mercat_results/'\
                                +str(READID)+".PATRIC_protein_run/"+str(READID)\
                                +".PATRIC_protein_summary.csv",header=0,sep=',')

    KEYS.update(list(MERCATres.iloc[:,0]))


ColNum = {}
ii = 0
for key in KEYS:
    ColNum[key] = ii
    ii = ii + 1


pickle.dump(KEYS,open('/lustre/scratch/tv349/AMR/KEYS/AA-'+Species+'-k'+str(K)+'KEYS.p','wb'))
pickle.dump(ColNum,open('/lustre/scratch/tv349/AMR/KEYS/AA-'+Species+'-k'+str(K)+'COLS.p','wb'))


## Run parallel
#if __name__ == '__main__':
#    p = Pool(NPrc)
#    p.map(ProcessChunk, list(range(NPrc)))

T3=datetime.datetime.now()
print('All done!')
print('K: '+str(K))
