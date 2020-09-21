#!/usr/bin/env SNP-env
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:45:29 2020

@author: taha
"""


import os
import sys
import numpy as np

INPUTFOLDER = sys.argv[1]
OUTPUTFOLDER = sys.argv[2]
REFPATH = sys.argv[3]
NCPU = sys.argv[4]
NPrc = int(sys.argv[5])
Thread = int(sys.argv[6])


FASTAFOLDER = INPUTFOLDER+'/'

os.chdir(OUTPUTFOLDER)


FILELIST = os.listdir(FASTAFOLDER)
FILELIST = sorted(FILELIST)




DataLen = len(FILELIST)
Q = int(np.ceil(DataLen / NPrc))
ChunkLen = len(range(Q*Thread,min(Q*(Thread+1),DataLen)))
for ii in range(Q*Thread,min(Q*(Thread+1),DataLen)):

    file = FILELIST[ii]
    print('')
    print('\n\n'+str(ii)+'/'+str(len(FILELIST))+'\n')
    COMMAND = "snippy --outdir "+file[:-4]+" --ctgs "+FASTAFOLDER+file+"  --ref "+REFPATH+"  --cpus "+str(NCPU)
    print(COMMAND)
    os.system(COMMAND)
