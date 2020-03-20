# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:03:05 2020
This code is used to describe information for first episode unmedicated SSD
@author: lenovo
"""
import pandas as pd

info_file = r'D:\WorkStation_2018\SZ_classification\Scale\cov_unmedicated_sp_and_hc_550.txt'
scale_file = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'

info = pd.read_csv(info_file)
scale = pd.read_excel(scale_file)


info_all = pd.merge(info, scale, left_on='folder', right_on='folder', how='inner')[['folder', 'diagnosis', 'age', 'sex', 'BPRS_Total', '病程月']]

info_descrb = info_all.groupby('diagnosis').describe()

sex_hc = info_all[info_all['diagnosis'] == 0]['sex'].value_counts()
sex_ssd = info_all[info_all['diagnosis'] == 1]['sex'].value_counts()

import pandas as pd
from matplotlib impo

   
