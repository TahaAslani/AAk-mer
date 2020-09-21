#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:30:19 2020

@author: taha
"""

from xlrd import open_workbook
import os
import ftplib

wb = open_workbook('/home/taha/Desktop/AMR/NeisseriaGonorrhoeae.xlsx')
for sheet in wb.sheets():
    number_of_rows = sheet.nrows
    number_of_columns = sheet.ncols

    genid = []
    for row in range(1, number_of_rows):
        value  = (sheet.cell(row,1).value)
        if value not in genid:
#            print(row/number_of_rows)
            genid.append(value)

genid = list(set(genid))

os.chdir('/home/taha/Desktop/AMR/FASTA')

address = "ftp.patricbrc.org"
with ftplib.FTP(address) as ftp:
    ftp.login()
    ftp.cwd("genomes")
    ii = 0
    for gid in genid:
        ii = ii+1
        print(str(ii)+'/'+str(len(genid)))
        f = gid + ".fna"
        ftp.cwd(gid)
        ftp.retrbinary('RETR ' + f, open(f, 'wb').write)
        ftp.cwd('../')
    ftp.quit()