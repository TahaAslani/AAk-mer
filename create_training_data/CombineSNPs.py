#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:46:41 2020

@author: taha
"""

import pandas as pd
import numpy as np
import os
import pickle
import sys

INPUTFOLDER = sys.argv[1]
OUTPUTFOLDER = sys.argv[2]

FASTAFOLDER = INPUTFOLDER
ADDRESS = OUTPUTFOLDER + '/'

def ExtractSNP(REF, ALT):
    OUT = []
    for x, y in zip(REF, ALT):
        if x != y:
            OUT.append(y)
    return OUT

NT2NUM = {'A':1, 'C':2, 'G':3, 'T':4}
NUM2NT = dict(map(reversed, NT2NUM.items()))
NT2OneHot = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1]}


os.chdir(ADDRESS)

#os.system('conda activate SNP-env')

#FILELIST = ['197.16208','197.15981','197.16338']
FILELIST = os.listdir(FASTAFOLDER)

print('Find union of columns')

COLS = set()
ROWS = []
for counter,filename in enumerate(FILELIST):
    
    file = filename[:-4]
    ROWS.append(file)
    print('\nUnion: '+str(counter)+'/'+str(len(FILELIST)))
    print(file)
    SNPgenome = pd.read_csv(ADDRESS+file+'/'+'snps.csv')
    for ii in range(SNPgenome.shape[0]):
        if SNPgenome.loc[ii,'TYPE'] == 'snp':
            COLS.add(SNPgenome.loc[ii,'POS'])

COLS = sorted(list(COLS))

ColNum = {}
ii = 0
for key in COLS:
    ColNum[key] = ii
    ii = ii + 1

RowNum = {}
ii = 0
for key in ROWS:
    RowNum[key] = ii
    ii = ii + 1


print('Creating SNP table...')
SNPnp = np.zeros([len(ROWS),len(COLS)],dtype=np.int8)
SNPnpOneHot = np.zeros([len(ROWS),4*len(COLS)],dtype=np.int8)
for counter,filename in enumerate(FILELIST):
    
    file = filename[:-4]
    print('\nAdd SNP: '+str(counter)+'/'+str(len(FILELIST)))
    print(file)
    SNPgenome = pd.read_csv(ADDRESS+file+'/'+'snps.csv')
    for ii in range(SNPgenome.shape[0]):
        if SNPgenome.loc[ii,'TYPE'] == 'snp':
#            print('\nAdd SNP: '+str(counter)+'/'+str(len(FILELIST)),str(ii)+'/'+str(SNPgenome.shape[0]))
            
            if len(SNPgenome.loc[ii,'ALT'])==1:
                SNPnp[RowNum[file],ColNum[SNPgenome.loc[ii,'POS']]] = NT2NUM[SNPgenome.loc[ii,'ALT']]
                POS = ColNum[SNPgenome.loc[ii,'POS']]
                SNPnpOneHot[RowNum[file],POS*4 : (POS+1)*4] = NT2OneHot[SNPgenome.loc[ii,'ALT']]
            
            # if alt is longer than one
            elif len(SNPgenome.loc[ii,'ALT'])>1:
                print(SNPgenome.loc[ii,'REF'])
                print(SNPgenome.loc[ii,'ALT'])
                ExtractedSNP = ExtractSNP(SNPgenome.loc[ii,'REF'], SNPgenome.loc[ii,'ALT'])
                if len(ExtractedSNP) == 1:
                    print(ExtractedSNP[0])
                    SNPnp[RowNum[file],ColNum[SNPgenome.loc[ii,'POS']]] = NT2NUM[ExtractedSNP[0]]
                    POS = ColNum[SNPgenome.loc[ii,'POS']]
                    SNPnpOneHot[RowNum[file],POS*4 : (POS+1)*4] = NT2OneHot[ExtractedSNP[0]]

SNP = pd.DataFrame(SNPnp, index = ROWS, columns = COLS)
COLSOneHot = []
for col in COLS:
    COLSOneHot.append(str(col)+'-A')
    COLSOneHot.append(str(col)+'-C')
    COLSOneHot.append(str(col)+'-G')
    COLSOneHot.append(str(col)+'-T')
SNPOneHot = pd.DataFrame(SNPnpOneHot, index = ROWS, columns = COLSOneHot) 
print('SNP table created')

    
print('Saving data...')
pickle.dump(SNP.astype(pd.SparseDtype("int")),open(ADDRESS+'SNPs.p','wb'),protocol=4)
pickle.dump(SNPOneHot.astype(pd.SparseDtype("int")),open(ADDRESS+'SNPOneHot.p','wb'),protocol=4)
#SNP.to_csv(ADDRESS+'SNPs.tsv',sep='\t')
#SNPOneHot.to_csv(ADDRESS+'SNPOneHot.tsv',sep='\t')
print('Data saved')

"""
print('Generating Logo...')
LOGO = pd.DataFrame(index = COLS, columns = list(NT2NUM.keys()))
for ii,pos in enumerate(LOGO.index):
    print(str(ii)+'/'+str(len(LOGO.index)))
    for NT in LOGO.columns:
        LOGO.loc[pos,NT] = np.sum(SNP.loc[:,pos]==NT2NUM[NT])

import matplotlib.pyplot as plt

plt.figure()
SNPCOUNT = np.sum(LOGO,axis=1)
plt.hist(SNPCOUNT,100)
plt.xlabel('Number of SNPs')
plt.ylabel('Frequency of positions with this number of SNPs')

plt.figure()
plt.hist(np.sum(LOGO !=0,axis=1),4,align='left')
plt.xlabel('Number of nucleotides after alteration')
plt.ylabel('Frequency of positions with this number')

print('Logo generated')

"""
