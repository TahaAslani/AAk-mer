#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:08:15 2020

@author: taha
"""

import os
import sys

INPUTFOLDER = sys.argv[1]
OUTPUTFILE = sys.argv[2] + '/PROTEIN.fa'

InputFiles = os.listdir(INPUTFOLDER)


filehandle = open(OUTPUTFILE, 'w')


for file in InputFiles:
    
    if file.endswith('.faa'):
        
#        print(file)
        
        with open(INPUTFOLDER+'/'+file) as fp:
            
            line = True
            while line:
                
                line = fp.readline()
                if line.startswith('>'):
#                    line = line.replace('|',']')
#                    line = line.replace(' ','-')
                    line = '>' + file + '$' + line[1:]
                    
                
                filehandle.write(line)


filehandle.close()
