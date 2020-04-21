# -*- coding: utf-8 -*-
"""This script is used to compare the mean FD between medicated and unmedicated subgroups with matched HCs.

"""


import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Statistics')
import pandas as pd
import numpy as np
from lc_chisqure import lc_chisqure
import scipy.stats as stats
import matplotlib.pyplot as plt

# Inputs
scales_whole = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'
results_ml = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy'
headmotionfile = r'D:\WorkStation_2018\SZ_classification\Scale\头动参数_1322.xlsx'
uidfile = r'D:\WorkStation_2018\SZ_classification\Scale\ID_1322.txt'
scale_206 = r'D:\WorkStation_2018\SZ_classification\Scale\SZ_NC_108_100.xlsx'

# Load
scales_whole = pd.read_excel(scales_whole)
headmotion = pd.read_excel(headmotionfile)
uid = pd.read_csv(uidfile, header=None)

# SZ filter
# 'mean FD_Power'
scales_1322 = pd.merge(scales_whole, uid, left_on='folder', right_on=0, how='inner')
scales_1322 = pd.merge(scales_1322, headmotion, left_on='folder', right_on='Subject ID', how='inner')

scales_sz = scales_1322[scales_1322['诊断'].isin([3])]
scales_hc = scales_1322[scales_1322['诊断'].isin([1])]
scales_sz_firstepisode = scales_sz[(scales_sz['首发']==1)]
scales_all = pd.concat([scales_hc, scales_sz])
