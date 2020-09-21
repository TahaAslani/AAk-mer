#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 01:00:26 2020

@author: taha
"""


import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
from Bio.SeqIO.FastaIO import FastaIterator
from Bio import SeqIO
from scipy import stats
import gc


FeaturePick = 50

KFOLD = 10


INPUTFOLDER = sys.argv[1]
cwd = sys.argv[2]
Species = sys.argv[3]
NCPU = sys.argv[4]

os.system('mkdir '+cwd+'/FeatureSelection')
OutputDir = cwd+'/FeatureSelection'
    
for IMPORTANCE_TYPE in ['SHAP']:
    
    print(IMPORTANCE_TYPE)
    
    PrintClassAcc = True
    
    
    # calculate the accuracy based on the ±2 dilution
    def accuracy_dilute(prdc,target):
        total = len(prdc);
        n = 0
        for i,j in zip(prdc,target):
            if j-1 <= i and i <= j+1:
                n += 1  
        return n/total
    
    
    
    
    KMERS = pickle.load(open(cwd+'/Features.p','rb'))
    
    
    Y_predict = []
    Y_test = []
    for CVcount in range(KFOLD):
        
        Y_predictFOLD = pickle.load(open(cwd+'/Y_predict-cv'+str(CVcount)+'.p', "rb"))
        Y_testFOLD = pickle.load(open(cwd+'/y_test-cv'+str(CVcount)+'.p', "rb"))
        
        Y_predict = np.concatenate([Y_predict,Y_predictFOLD])
        Y_test = np.concatenate([Y_test,Y_testFOLD])
    
    YList = sorted(list(set(Y_test)))
    
#    CONF = np.zeros([len(YList),len(YList)])
#    for ii in range(len(Y_predict)):
#        CONF[YList.index(Y_predict[ii]),YList.index(Y_test[ii])] += 1
#        
#    X = []
#    Y = []
#    R = []
#    for ii in YList:
#        for jj in YList:
#            X.append(ii)
#            Y.append(jj)
#            R.append(CONF[YList.index(ii),YList.index(jj)])
        
#    plt.figure()
#    plt.scatter(X,Y,s=R)
    
    x = np.array(range(int(np.min(Y_test)),1+int(np.max(Y_test))))
    plt.plot(x,x,color='g')
    
    Z1 = np.polyfit(Y_test,Y_predict,1)
    P1 = np.poly1d(Z1)
    R2 = metrics.r2_score(Y_predict,P1(Y_test))
    plt.plot(Y_test,P1(Y_test),color='y')
    
    
    plt.plot(x,x-1,color='r')
    plt.plot(x,x+1,color='r')
    
    plt.xlabel('Actual value (log2 scale)')
    plt.ylabel('Predicted value (log2 scale)')
    plt.title('. Predicted vs actual values. Accuracy:'+\
              str(np.round(accuracy_dilute(Y_predict,Y_test),4)))
    plt.legend(['Perfect prediction','First order regression (R^2 score:'+str(np.round(R2,2))+')','Two-fold dilution'])
    
    plt.savefig(OutputDir+"/Scatter.jpg",format='jpg')
    
    
    ActualValues = list(set(Y_test))
    ActualValues.sort()
    NumberOfValues = len(ActualValues)
    MICDIC = {}
    for VAL in ActualValues:
        MICDIC[VAL] = ActualValues.index(VAL)
    
    
    FIG = plt.figure()
    DATA = [[] for _ in range(NumberOfValues)]
    ACTUALDATA = [[] for _ in range(NumberOfValues)]
    
    for ii in range(len(Y_test)):
        predic = Y_predict[ii]
        test = Y_test[ii]
        
        DATA[MICDIC[test]].append(predic)
        ACTUALDATA[MICDIC[test]].append(test)
        
    ACCclass = []
    NumberDataClass = []
    for ii in range(NumberOfValues):
        ACCclass.append(accuracy_dilute(DATA[ii],ACTUALDATA[ii]))
        NumberDataClass.append(len(DATA[ii]))
    
    plt.violinplot(DATA,showmeans=True,showextrema=True,positions=ActualValues)
    plt.plot(x,x,color='g')
    
    Z1 = np.polyfit(Y_test,Y_predict,1)
    P1 = np.poly1d(Z1)
    R2 = metrics.r2_score(Y_predict,P1(Y_test))
    plt.plot(Y_test,P1(Y_test),color='y')
    
    plt.plot(x,x-1,color='r')
    plt.plot(x,x+1,color='r')
    plt.xlabel('Actual value (log2 scale)')
    plt.ylabel('Predicted value (log2 scale)')
    plt.title('. Predicted vs actual values. Accuracy:'+\
              str(np.round(accuracy_dilute(Y_predict,Y_test),4)))
    plt.legend(['Perfect prediction','First order regression (R^2 score:'+str(np.round(R2,2))+')','Two-fold dilution'])
    
    if PrintClassAcc:
            
        for ii in range(len(ACCclass)):
            STR = str(np.round(ACCclass[ii],2))
            STR2 = '('+str(NumberDataClass[ii])+')'
            plt.text(ActualValues[ii]-0.2,np.min(np.min(DATA))-1,STR)
            plt.text(ActualValues[ii]-0.2,np.min(np.min(DATA))-1.3,STR2)
    
    
    plt.savefig(OutputDir+"/Violin.jpg",format='jpg')
    pickle.dump(FIG, open(OutputDir+"/Violin.pickle",'wb'),protocol=3)
    
    
    # Feat table
    CV = []
    
    for CVcount in range(KFOLD):
        
        model = pickle.load( open( cwd+"/GXMODEL-cv"+str(CVcount)+".dat", "rb" ) )
        
        if IMPORTANCE_TYPE == 'SHAP':
            FeatureOrder = pickle.load(open(cwd+'/FeatureOrder-cv'+str(CVcount)+'.p', "rb"))
            
        else:
            XGGAIN = model.get_score(importance_type=IMPORTANCE_TYPE)
            
            SORTEDLIST = [k for k, v in sorted(XGGAIN.items(),reverse=True, key=lambda item: item[1])]
            FeatureOrder = []
            for ii in SORTEDLIST:
                FeatureOrder.append(int(ii[1:]))
        
        NumberOfKmers = 0
        counter = 0
        ImportantKmers = []
        while NumberOfKmers < FeaturePick:

            ImportantKmers.append(FeatureOrder[counter])
            NumberOfKmers = NumberOfKmers + 1
            counter = counter + 1
           
            #if we don't have more features
            if NumberOfKmers==len(FeatureOrder):
                break
        
        CV.append(ImportantKmers)
    
    
    
    IND = set()
    SUM = 0
    for ii in range(len(CV)):
        IND.update(CV[ii])
        SUM = SUM + len(CV[ii])
    
    ALLFEATURES = pickle.load(open(cwd+'/Features.p','rb'))

    
    DIC = {}
    for feat in IND:
        DIC[feat] = ALLFEATURES[feat]
        
    
    PresenceColName = 'PresenceInTop'+str(FeaturePick)
    COLs = [PresenceColName]
    
    
    DF = pd.DataFrame(index=DIC.values(),columns=COLs)
    DF[PresenceColName] = 0
    for ii in range(len(CV)):
        for feat in DIC.keys():
    
            
            if feat in CV[ii]:
                DF.loc[DIC[feat],PresenceColName] = DF.loc[DIC[feat],PresenceColName] + 1
    #        else:
    #            DF.loc[DIC[feat],ii] = np.nan
    #
    
#    plt.close('all')
    DF = DF.sort_values(by=[PresenceColName],ascending=False)
    DF.to_csv(OutputDir+'/FEATTABLE-'+IMPORTANCE_TYPE+'.csv')
    
    #DFN = DF.replace({0: np.nan})
    #msno.matrix(DFN.set_index(DFN.index))
    #seaborn.heatmap(DF, linewidths=2)
    
    
    
    #result_handle = NCBIWWW.qblast("blastn", "nt", 'GGTTCTGTTGCAACA',megablast=True\
    #                               ,word_size=7,expect=1000,hitlist_size=100,\
    #                               gapcosts='5 2',genetic_code=1, filter='Staphylococcus aureus')
    #blast_record = NCBIXML.read(result_handle)
    #
    #for alignment in blast_record.alignments:
    #    
    #    print()
    #    print(alignment.title)
    #    
    #    
    #    for hsp in alignment.hsps:
    #        print(hsp.sbjct_start,hsp.sbjct_end)




###############################################################################################
# Find biological meaning of the features


###########################################
df = pd.DataFrame()

FileCounter = 0
for file in sorted(os.listdir(INPUTFOLDER)):
    if file.endswith(".p"):
        df_temp = pickle.load(open(INPUTFOLDER+'/'+file, "rb"))
        df = pd.concat([df, df_temp])

        del df_temp
        gc.collect()

        FileCounter = FileCounter + 1
        print(str(FileCounter)+' file(s) loaded!')
###########################################



FOLDER = sys.argv[2]

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

    if file.startswith('FEATTABLE-SHAP.csv'):
        
        print(cwd+'/'+file)
        
        DEST = cwd+'/'+file.split('.')[0]
        os.system('mkdir '+DEST)
        
        DF = pd.read_csv(cwd+'/'+file)
        
        # Add coumns:
        DF['P-value'] = np.nan
        DF['Nwith'] = np.nan
        DF['Nwithout'] = np.nan
        DF['meanWith'] = np.nan
        DF['meanWithout'] = np.nan
        DF['STDwith'] = np.nan
        DF['STDwithout'] = np.nan


        
        
        
        for ii in range(DF.shape[0]):
            
            print('Progress: '+str(ii)+' out of '+str(DF.shape[0]))
            
            Query = DF.iloc[ii,0]
            Presence = DF.iloc[ii,1]
            
            f = open(cwd+'/QUERY.fa','w')
            f.write('>'+Query+'\n'+Query)
            f.close()
            
            commandBLAST = 'blastp -db /pylon5/br5phhp/tv349/AMR/AADB/'+Species+'.fa -query '\
                +cwd+'/QUERY.fa -out '+DEST+'/'+str(Presence)+'-'+Query+\
                    '.tsv -outfmt 7 -num_threads '+str(NCPU)+\
                    ' -word_size 2 -gapopen 9 -gapextend 1 -matrix PAM30 -window_size 40 -comp_based_stats 0 -evalue 200000 -task blastp-short'
            os.system(commandBLAST)
            
            COL = ['query', 'subject', 'identity', 'alignment length',\
                   'mismatches', 'gap opens', 'q. start', 'q. end', 's. start',\
                       's. end', 'evalue', 'bit score']
            BLASTres = pd.read_csv(DEST+'/'+str(Presence)+'-'+Query+'.tsv',sep='\t', names=COL)
            
            for jj in range(BLASTres.shape[0]):
                
                if BLASTres.loc[jj,'query'].startswith('#'):
                    continue
                if BLASTres.loc[jj,'mismatches']>0 or\
                    BLASTres.loc[jj,'gap opens']>0 or\
                        BLASTres.loc[jj,'identity']<100 or\
                            BLASTres.loc[jj,'alignment length'] != len(BLASTres.loc[jj,'query']):
                    continue
                
                SUBJECT = BLASTres.loc[jj,'subject']
                
                PATRICID = SUBJECT.split('.peg.')[0]
                PATRICID = PATRICID.split(':')[1]
                
                Header = SUBJECT.replace('fig:','fig|')
                
            

                OUT = FindGene(PATRICID, Header)
                for key in OUT.keys():
                    BLASTres.loc[jj,key] = OUT[key]
            
            
            ToDisk = BLASTres.drop(['identity', 'alignment length','mismatches',\
                                    'gap opens', 'q. start', 'q. end','evalue',\
                                        'bit score'], axis=1)
            ToDisk.to_csv(DEST+'/'+str(Presence)+'-'+Query+'.csv',index=None)
            os.remove(DEST+'/'+str(Presence)+'-'+Query+'.tsv')
            
            


            featName = Query
            feat_mask = df[featName].astype('bool')

            DF.loc[ii,'Nwith'] = np.sum(feat_mask)
            DF.loc[ii,'Nwithout'] = np.sum(~feat_mask)

            MICwit = np.power(2,df.loc[feat_mask,'MIC'].astype('float').values)
            MICwithout = np.power(2,df.loc[~feat_mask,'MIC'].astype('float').values)

            DF.loc[ii,'meanWith'] = np.mean(MICwit)
            DF.loc[ii,'STDwith'] = np.std(MICwit)

            DF.loc[ii,'meanWithout'] = np.mean(MICwithout)
            DF.loc[ii,'STDwithout'] = np.std(MICwithout)

            DF.loc[ii,'P-value'] = stats.kruskal(MICwit,MICwithout)[1]

        DF.to_csv(cwd+'/'+file)

