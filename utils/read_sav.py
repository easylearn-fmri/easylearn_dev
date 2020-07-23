# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 09:26:39 2019

@author: lenovo
"""
import pandas as pd
s = pd.read_excel('02_18大表(REST).xlsx')
s1 = s[['folder', '诊断']]

s_hc = s1[s1['诊断'] == 1]
s_hc['folder'].to_excel('HC.xlsx', header=False, index=False)

s_hc = s1[s1['诊断'] == 2]
s_hc['folder'].to_excel('MDD.xlsx', header=False, index=False)

s_hc = s1[s1['诊断'] == 3]
s_hc['folder'].to_excel('SZ.xlsx', header=False, index=False)

s_hc = s1[s1['诊断'] == 4]
s_hc['folder'].to_excel('BD.xlsx', header=False, index=False)
