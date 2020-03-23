# -*- coding: utf-8 -*-
"""
This script is used to cheack data quality, especially the head motion
"""
import pandas as pd
import numpy as np

# Inputs
headmotionfile = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\10-24´ó±í.xlsx'
uidfile = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\ID_1322.txt'

# Load
headmotion = pd.read_excel(headmotionfile)
uid = pd.read_csv(uidfile, header=None)

# Merger
screened_headmotion = pd.merge(headmotion, uid, left_on='Subject ID', right_on=0, how='inner')
colname = list(screened_headmotion.columns)
print(colname)
rigbody_rotation = screened_headmotion[['Subject ID','max(abs(Tx))', 'max(abs(Ty))', 'max(abs(Tz))',
                                     'max(abs(Rx))', 'max(abs(Ry))', 'max(abs(Rz))']]
mFD = screened_headmotion[['Subject ID','mean FD_Power']]
Percent_of_great_FD = screened_headmotion[['Subject ID','Percent of FD_Power>0.2']]

print(np.mean(mFD))
print(np.mean(Percent_of_great_FD))
