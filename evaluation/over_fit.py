#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:51:36 2020

@author: taha
"""


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import pickle
import scipy.stats


FOLDERs = ['/pylon5/br5phhp/tv349/AMR/OUTC',
           '/pylon5/br5phhp/tv349/AMR/OUTC',
           '/pylon5/br5phhp/tv349/AMR/OUTC',
           '/pylon5/br5phhp/tv349/AMR/OUTC',
           '/pylon5/br5phhp/tv349/AMR/OUTP',
           '/pylon5/br5phhp/tv349/AMR/OUTP',
           '/pylon5/br5phhp/tv349/AMR/OUTP',
           '/pylon5/br5phhp/tv349/AMR/OUTG',
           '/pylon5/br5phhp/tv349/AMR/OUTSNP',
           '/pylon5/br5phhp/tv349/AMR/OUTGSNP']

NumberOfKmerData = 7

Klist = [8,9,10,11,3,4,5]

LEGEND = ['NT 8-mers','NT 9-mers','NT 10-mers','NT 11-mers'\
              ,'AA 3-mers', 'AA 4-mers','AA 5-mers',\
                  'Gene content','SNP','Gene content + SNP']


COLOR = ['aqua','deepskyblue','dodgerblue','blue',\
         'palegreen','limegreen','darkgreen',\
                'gray','yellow','saddlebrown']

WIDTH = 0.08

SPE = 'CampylobacterJejuni'

ABLIST = ['erythromycin', 'azithromycin', 'gentamicin', 'clindamycin', 'telithromycin', 'ciprofloxacin', 'nalidixicacid', 'tetracycline', 'florfenicol']





ANTIB = []

for ii in range(len(ABLIST)):
    ANTIB.append(SPE+'-'+ABLIST[ii])




OUTIMAGE = SPE


LINES = []
for ii in range(len(COLOR)):
    LINES.append(Line2D([0], [0], color=COLOR[ii], lw=4))


ANTIB.sort(reverse = True)



fig_handle = plt.figure()


CleanName = []

DATAkw = []
YTick= []
for COUNTER in range(len(FOLDERs)):


    cwd = FOLDERs[COUNTER]

    
    print('Data'+str(COUNTER))


    
    DATA1 = []
    
    ALL = []
    for ii in ANTIB:
        if COUNTER < NumberOfKmerData:
            k = Klist[COUNTER]
            try:
                ACC = pd.read_csv(cwd+'/'+ii+'/k'+str(k)+'/ACCU.txt',header = None)
            except:
                print(ii+"Doesn't exist")
                continue
        else:
            try:
                ACC = pd.read_csv(cwd+'/'+ii+'/ACCU.txt',header = None)
            except:
                print(ii+"Doesn't exist")
                continue

        if ACC.shape[0] != 10:
       	    #raise Exception('Shape is not 10: '+cwd)
            print(('Warning: Shape is not 10: '+cwd))   

        ALL += list(list(ACC.values.transpose())[0])
    print('All',np.mean(ALL),np.std(ALL),len(ALL))

    DATAkw.append(ALL)
    DATA1.append(ALL)
    CleanName.append(LEGEND[COUNTER]+' - Test Accuracy')
    CleanName.append(LEGEND[COUNTER]+' - Train Accuracy')
    
    
    POS = -COUNTER*WIDTH+np.array(range(len(DATA1)))
    box1 = plt.boxplot(DATA1,vert=False,showmeans=True,meanline=True,showfliers=False,\
                positions=POS,widths = WIDTH/3,\
                patch_artist=True,whis='range')
    plt.setp(box1["boxes"],facecolor=COLOR[COUNTER])
    
    YTick.append(POS)    
    
    
    DATA1 = []
    
    ALL = []
    for ii in ANTIB:
        if COUNTER < NumberOfKmerData:
            k = Klist[COUNTER]
            try:
                ACC = pd.read_csv(cwd+'/'+ii+'/k'+str(k)+'/ACCUtrain.txt',header = None)
            except:
                print(ii+"Doesn't exist")
                continue
        else:
            try:
                ACC = pd.read_csv(cwd+'/'+ii+'/ACCUtrain.txt',header = None)
            except:
                print(ii+"Doesn't exist")
                continue
        ALL += list(list(ACC.values.transpose())[0])
    print('All',np.mean(ALL),np.std(ALL),len(ALL))
      
    DATA1.append(ALL)
    
    POS2 = -WIDTH/3-COUNTER*WIDTH+np.array(range(len(DATA1)))
    box1 = plt.boxplot(DATA1,vert=False,showmeans=True,meanline=True,showfliers=False,\
                positions=POS2,widths = WIDTH/3,\
                patch_artist=True,whis='range')
    plt.setp(box1["boxes"],facecolor=COLOR[COUNTER])
    YTick.append(POS2)








plt.title(SPE)
plt.xlabel('Distribution of accuracies of folds of cross-validation', weight='bold')



plt.yticks(YTick,(CleanName))

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

plt.grid()
plt.tight_layout()
#plt.xlim([0.88,1])


custom_lines = LINES
plt.legend(custom_lines[:len(FOLDERs)],LEGEND[:len(FOLDERs)])

plt.savefig('/pylon5/br5phhp/tv349/AMR/CompareBox/'+OUTIMAGE+'OF'+'.png')
plt.savefig('/pylon5/br5phhp/tv349/AMR/CompareBox/'+OUTIMAGE+'OF'+'.pdf')
pickle.dump(fig_handle, open('/pylon5/br5phhp/tv349/AMR/CompareBox/'+OUTIMAGE+'OF'+'.pickle', 'wb'))






#KW TEST


DF = pd.DataFrame(index=LEGEND, columns=LEGEND)
for ii,test1 in enumerate(LEGEND):

    ACC1 = DATAkw[ii]

    jj = ii + 1
    for test2 in LEGEND[ii+1:]:

        ACC2 = DATAkw[jj]
        DF.loc[test1,test2] = scipy.stats.kruskal(ACC1,ACC2).pvalue
        print(ii,jj)
        jj = jj + 1

DF.to_csv('/pylon5/br5phhp/tv349/AMR/CompareBox/KW-'+OUTIMAGE+'.csv')
