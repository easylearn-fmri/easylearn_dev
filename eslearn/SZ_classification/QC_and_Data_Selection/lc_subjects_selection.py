# -*- coding: utf-8 -*-
"""
This script is used to select the subjects with good quality (mean FD, percentages of greater FD, rigid motion).
Then matching SZ and HC based on the age, sex and headmotion.
Note that: these 1322 subjects are already selected by rigid motion criteria: one voxel.
All selected subjects's ID will save to D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\selected_sub.xlsx
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Statistics')
import pandas as pd
import numpy as np
from lc_chisqure import lc_chisqure
import scipy.stats as stats
import matplotlib.pyplot as plt

# Inputs
scales_whole = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\10-24大表.xlsx'
headmotionfile = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\头动参数_1322.xlsx'
uidfile = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\ID_1322.txt'
scale_206 = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\SZ_NC_108_100.xlsx'

# Load
scales_whole = pd.read_excel(scales_whole)
headmotion = pd.read_excel(headmotionfile)
uid = pd.read_csv(uidfile, header=None)

# SZ filter
scales_1322 = pd.merge(scales_whole, uid, left_on='folder', right_on=0, how='inner')
scales_sz = scales_1322[scales_1322['诊断'].isin([3])]
scales_hc = scales_1322[scales_1322['诊断'].isin([1])]
scales_sz_firstepisode = scales_sz[(scales_sz['首发']==1)]
#scales_sz_firstepisode = scales_sz[(scales_sz['用药'].isin([0]))]
scales_all = pd.concat([scales_hc, scales_sz])

# Headmotion filter
scales_all_headmotionfilter = pd.merge(headmotion, scales_all, left_on='Subject ID', right_on='folder', how='inner')
colname = list(scales_all_headmotionfilter.columns)
motion = scales_all_headmotionfilter.iloc[:,[1,2,3,4,5,6]]
mFD = scales_all_headmotionfilter[['Subject ID','mean FD_Power']]
Percent_of_great_FD = scales_all_headmotionfilter[['Subject ID','Percent of FD_Power>0.2']]

# scales_all_headmotionfilter = scales_all_headmotionfilter[(scales_all_headmotionfilter['mean FD_Power'] <= 0.2) \
# 								& (scales_all_headmotionfilter['Percent of FD_Power>0.2'] <= 0.2)]

# Number matching (Let Number of HC = Number of SZ)
scales_all_headmotionfilter = scales_all_headmotionfilter.drop(scales_all_headmotionfilter[(scales_all_headmotionfilter['诊断'] == 1)].index[0:114])
print(scales_all_headmotionfilter[(scales_all_headmotionfilter['诊断'] == 1)].index)        
# scales_all_headmotionfilter.to_excel(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\scale_cnn.xlsx')

# Matching based on age, sex and headmotion
age_hc = scales_all_headmotionfilter['年龄'][scales_all_headmotionfilter['诊断']==1]
age_sz = scales_all_headmotionfilter['年龄'][scales_all_headmotionfilter['诊断']==3]
mage_hc = np.mean(age_hc)
age_hc = scales_all_headmotionfilter['年龄'][(scales_all_headmotionfilter['诊断']==1) \
                                    & (scales_all_headmotionfilter['年龄'] < 42)]
t, p = stats.ttest_ind(age_hc, age_sz)
print(f'p_age = {p}\n')
scales_all_headmotionfilter = scales_all_headmotionfilter[((scales_all_headmotionfilter['诊断']==1) \
                                    & (scales_all_headmotionfilter['年龄'] < 42)) | (scales_all_headmotionfilter['诊断']==3)]


sex_hc = scales_all_headmotionfilter['性别'][scales_all_headmotionfilter['诊断']==1]
sex_sz = scales_all_headmotionfilter['性别'][scales_all_headmotionfilter['诊断']==3]
numsex_hc = [np.sum(sex_hc==1), np.sum(sex_hc==2)]
numsex_sz = [np.sum(sex_sz==1), np.sum(sex_sz==2)]
obs = [np.sum(sex_hc==1), np.sum(sex_sz==1)]
tt = [len(sex_hc), len(sex_sz)]
chivalue, chip = lc_chisqure(obs, tt)
print(f'p_sex = {chip}\n')

mFD_hc = scales_all_headmotionfilter['mean FD_Power'][scales_all_headmotionfilter['诊断']==1]
mFD_sz = scales_all_headmotionfilter['mean FD_Power'][scales_all_headmotionfilter['诊断']==3]
t, p = stats.ttest_ind(mFD_hc, mFD_sz)
print(f'p_mFD = {p}\n')

# save
# scales_all_headmotionfilter['Subject ID'].to_csv('D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\selected_550.txt',index=False, header=False)
print(f'Totle selected participants is {np.shape(scales_all_headmotionfilter)[0]}')
print(f"number of HC = {np.sum(scales_all_headmotionfilter['诊断']==1)}")
print(f"number of SZ = {np.sum(scales_all_headmotionfilter['诊断']==3)}")


# -------------------------------206--------------------------------------------
scale_206 = pd.read_excel(scale_206)

age_g1 = scale_206['age'][scale_206['group']==1]
age_g2 = scale_206['age'][scale_206['group']==2]
t, p = stats.ttest_ind(age_g1, age_g2)
#plt.hist(age_g1, bins=50)
#plt.hist(age_g2, bins=50)
#plt.legend(['sz','hc'])
#plt.show()


num1_g1 = np.sum(scale_206['sex'][scale_206['group']==1])
num0_g1 = np.sum(scale_206['sex'][scale_206['group']==1] ==0)
num1_g2 = np.sum(scale_206['sex'][scale_206['group']==2])
num0_g2 = np.sum(scale_206['sex'][scale_206['group']==2] ==0)
obs = [num1_g1, num1_g2]
tt = [np.sum(scale_206['group']==1), np.sum(scale_206['group']==2)]
chivalue, chip = lc_chisqure(obs, tt)

#plt.subplot(121)
#plt.pie([num1_g1,num0_g1])
#plt.legend(['1','0'])
#plt.title('g1')
#
#plt.subplot(122)
#plt.pie([num1_g2,num0_g2])
#plt.legend(['1','0'])
#plt.title('g2')
#plt.show()