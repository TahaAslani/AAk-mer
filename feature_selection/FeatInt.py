#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 02:21:38 2020

@author: taha
"""

from Bio import SeqIO
import os
import pandas as pd
import numpy as np
from Bio.SeqIO.FastaIO import FastaIterator

FOLDER = '/pylon5/br5phhp/tv349/AMR/HRESGSNP/CampylobaterJejuni-azithromycin'
Species = 'CampylobacterJejuni'


Refrence = '/pylon5/br5phhp/tv349/AMR/REF/'+Species+'.gbff'

NumberofImportantFeat = 20

def FindSNP(Refrence, Position):
 
    VICTORY = False
    
    for gb_record in SeqIO.parse(open(Refrence,"r"), "genbank") :
        for feature in gb_record.features:
            if feature.type == 'CDS':
                if feature.location.__contains__(Position):
                    VICTORY = True                    
                    if 'gene' in feature.qualifiers.keys():
                        GENE = feature.qualifiers['gene']
                    else:
                        GENE = ''
                    if 'product' in feature.qualifiers.keys():
                        PRODUCT = feature.qualifiers['product']
                    else:
                        PRODUCT = ''
                    if 'translation' in feature.qualifiers.keys():
                        TRANSLATION = feature.qualifiers['translation']
                    else:
                        TRANSLATION = ''
    if VICTORY:
        return GENE, PRODUCT, TRANSLATION
    else:
        return 'CDS not found','CDS not found'

def FindGene(PATRICID, Header):
 
    OUT = dict()
    
    SPGENE = pd.read_csv('/pylon5/br5phhp/tv349/AMR/PATRIC/SPGENE/'+PATRICID+'.PATRIC.spgene.tab',sep='\t')
    LocalPos = SPGENE.index[SPGENE['patric_id'] == Header].tolist()
    # if the sequence exists here:
    OUTSPGENE = dict()
    if len(LocalPos) == 1:
        OUTSPGENE = (SPGENE.loc[LocalPos,['gene', 'product', 'property', 'function']]).to_dict('records')[0]
    
    FEATURES = pd.read_csv('/pylon5/br5phhp/tv349/AMR/PATRIC/FEATURES/'+PATRICID+'.PATRIC.features.tab',sep='\t')
    LocalPos = FEATURES.index[FEATURES['patric_id'] == Header].tolist()
    OUTFEATURES = dict()
    if len(LocalPos) == 1:
        OUTFEATURES = (FEATURES.loc[LocalPos,['gene','product']]).to_dict('records')[0]
        
    OUT = {**OUTFEATURES, **OUTSPGENE}
    
    # Get sequence
    with open("/pylon5/br5phhp/tv349/AMR/PATRIC/PROTEIN/"+PATRICID+".PATRIC.faa") as handle:
        for record in FastaIterator(handle):
            if record.id == Header:
                AAseq = str(record.seq)
    
    OUT['translation'] = AAseq
    
    return OUT


cwd = FOLDER+'/FeatureSelection'

FILES = os.listdir(cwd)
for file in FILES:

    if file.startswith('FEATTABLE'):
        
        print(cwd+'/'+file)
        
        DF = pd.read_csv(cwd+'/'+file)
        
        # Add coumns:
        DF['gene'] = np.nan
        DF['product'] = np.nan
        DF['property'] = np.nan
        DF['function'] = np.nan
        DF['translation'] = np.nan

        
        
        
        for ii in range(NumberofImportantFeat):
            
            featName = DF.iloc[ii,0]
            
            if featName.startswith('SNP-'):
                Position = int(''.join([i for i in featName if i.isdigit()]))
                GENE, PRODUCT, TRANSLATION = FindSNP(Refrence, Position)
                
                DF.loc[ii,'gene'] = GENE
                DF.loc[ii,'product'] = PRODUCT
                DF.loc[ii,'translation'] = TRANSLATION
                
            elif featName.startswith('G-'):
                FileName, Header = featName.split('$')
                FileName = FileName[2:] #Remove G-
                PATRICID = FileName.split('.PATRIC')[0]
                
                OUT = FindGene(PATRICID, Header)
                for key in OUT.keys():
                    DF.loc[ii,key] = OUT[key]
                # Read seq
            
        DF.to_csv(cwd+'/'+file)

