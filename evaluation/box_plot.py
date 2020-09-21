#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:54:44 2019

@author: taha
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import pickle


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





all_acc = pd.DataFrame(index=LEGEND,columns=[SPE])

OUTIMAGE = 'Figure'


LINES = []
for ii in range(len(COLOR)):
    LINES.append(Line2D([0], [0], color=COLOR[ii], lw=4))


ANTIB.sort(reverse = True)



fig_handle = plt.figure()

for COUNTER in range(len(FOLDERs)):


    cwd = FOLDERs[COUNTER]
    
    
    print('Data'+str(COUNTER))
    CleanName = []
    DATA1 = []
    HO1 = []
    
    for ii in ANTIB:
        
        if COUNTER < NumberOfKmerData:
            k = Klist[COUNTER]
            try:
                ACC = pd.read_csv(cwd+'/'+ii+'/k'+str(k)+'/ACCU.txt',header = None)
                HO = pd.read_csv(cwd+'/'+ii+'/k'+str(k)+'/Held-Out-Eval/ACCU.txt',header = None)
            except:
                print(ii+"Doesn't exist")
                continue
        else:
            try:
                ACC = pd.read_csv(cwd+'/'+ii+'/ACCU.txt',header = None)
                HO = pd.read_csv(cwd+'/'+ii+'/Held-Out-Eval/ACCU.txt',header = None)
            except:
                print(ii+"Doesn't exist")
                continue
        
        
        DATA1.append(list(ACC.values.transpose())[0])
        HO1.append(HO.values[0][0])
        
        CleanName.append(ii.split('-')[1])
        
        print(ii,np.mean(ACC.values),np.std(ACC.values),len(ACC.values))
    
    
    ALL = []
    for ii in ANTIB:
        if COUNTER < NumberOfKmerData:
            k = Klist[COUNTER]
            try:
                ACC = pd.read_csv(cwd+'/'+ii+'/k'+str(k)+'/ACCU.txt',header = None)
                HO = pd.read_csv(cwd+'/'+ii+'/k'+str(k)+'/Held-Out-Eval/ACCU.txt',header = None)
            except:
                print(ii+"Doesn't exist")
                continue
        else:
            try:
                ACC = pd.read_csv(cwd+'/'+ii+'/ACCU.txt',header = None)
                HO = pd.read_csv(cwd+'/'+ii+'/Held-Out-Eval/ACCU.txt',header = None)
            except:
                print(ii+"Doesn't exist")
                continue
        ALL += list(list(ACC.values.transpose())[0])
    print('All',np.mean(ALL),np.std(ALL),len(ALL))
      
    DATA1.append(ALL)
    CleanName.append('ALL')
    
    
    box1 = plt.boxplot(DATA1,vert=False,showmeans=True,meanline=True,showfliers=False,\
                positions=-COUNTER*WIDTH+np.array(range(len(DATA1))),widths = WIDTH,\
                patch_artist=True,whis='range',zorder = 0)
    plt.setp(box1["boxes"],facecolor=COLOR[COUNTER])
    
    plt.plot(HO1,-COUNTER*WIDTH+np.array(range(len(DATA1)-1)),'x',color='k')

    all_acc.loc[LEGEND[COUNTER],SPE] = str(np.round(np.mean(ALL),3))+' ('+str(np.round(np.std(ALL),3))+')'

all_acc.to_csv('/pylon5/br5phhp/tv349/AMR/CompareBox/All-'+SPE+'.csv')





plt.title(SPE, weight='bold')

plt.xlabel('Distribution of accuracies of folds of cross-validation', weight='bold')

plt.ylabel('Anti-biotic name', weight='bold')

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

plt.yticks(np.array(range(len(CleanName))),(CleanName),)
plt.grid()
plt.tight_layout()
#plt.xlim([0.88,1])


custom_lines = LINES
plt.legend(custom_lines[:len(FOLDERs)],LEGEND[:len(FOLDERs)])

plt.savefig('/pylon5/br5phhp/tv349/AMR/CompareBox/'+OUTIMAGE+SPE+'.png')
plt.savefig('/pylon5/br5phhp/tv349/AMR/CompareBox/'+OUTIMAGE+SPE+'.pdf')
pickle.dump(fig_handle, open('/pylon5/br5phhp/tv349/AMR/CompareBox/'+OUTIMAGE+SPE+'.pickle', 'wb'),protocol=2)
print('Figures are saved in /pylon5/br5phhp/tv349/AMR/CompareBox/')
