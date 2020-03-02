# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:57:57 2019

@author: lenovo
"""

import numpy as np
import pandas as pd

# roi 2 net
with open(r'F:\Data\ASD\ROI_160.txt', 'r') as f:
    info = f.readlines()

info = pd.Series(info)

# roi
roi = info.str.findall('\d.*[a-zA-Z]*\d')
roi = pd.Series([rn[0] for rn in roi])
roi = roi.str.findall('[a-zA-Z].*[a-zA-Z]')

s=[]
for roi_ in roi:
    s.append([ss for ss in roi_ if ss.strip() != ''])   
s = [s[0]+ ' ' + s[1] if len(s) == 2 else s[0] for s in s] 
roi = pd.Series(s)

# net
net = pd.Series([info_.split(' ')[-1].strip() for info_ in info])


# excel
exl = pd.read_csv(r'F:\Data\ASD\dos160_labels.csv')
roi_exl = exl.iloc[:,1]

roi_exl=list(roi_exl)
roi = list(roi)
ll = [roi_exl.index(roi_) for roi_ in roi]
net_exl = net[ll]
net_exl.to_excel(r'F:\Data\ASD\network.xlsx')
