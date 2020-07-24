# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:35:09 2018
画热图
@author: lenovo
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
##=====================================================================

# input
corr=pd.read_excel(r'D:\workstation_b\彦鸽姐\20190927\r_dti1.xlsx',header=None,index=None)
pValue=pd.read_excel(r'D:\workstation_b\彦鸽姐\20190927\p_dti1.xlsx',header=None,index=None)
x=pd.read_excel(r'D:\workstation_b\彦鸽姐\20190927\DTI(1).xlsx')
if_save_figure=0
#==================================================================
corr[pValue.isnull()]=None
pValue[pValue.isnull()]=None
mask=pValue>0.037
# =============================================================================
# #调整顺序
#columns=list(x.columns)
#col_index=[10,9,11,5,12,8,6,7,2,3,4,1,17,18,19,20,21,22]
#col=[columns[i] for i in col_index]
col=list(list(x.columns))[3:]
# =============================================================================
# colormap
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
#f, (ax1) = plt.subplots(figsize=(20,20),nrows=1)
f, (ax1) = plt.subplots(nrows=1)

#sns.heatmap(x, annot=True, ax=ax1,cmap='rainbow',center=0)#cmap='rainbow'
sns.heatmap(corr,ax=ax1,
            annot=True,annot_kws={'size':6,'weight':'normal', 'color':"k"},fmt='.3f',
            cmap='RdBu_r',
            linewidths = 0.05, linecolor= 'k',
            mask=mask)


ax1.set_title('')
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_xticklabels(col,size=9)
ax1.set_yticklabels(col,size=9)

## 设置选中，以及方位
label_x = ax1.get_xticklabels()
label_y = ax1.get_yticklabels()
plt.setp(label_x, rotation=15, horizontalalignment='right')
plt.setp(label_y, rotation=0, horizontalalignment='right')
plt.show()

# save
if if_save_figure:
    plt.savefig(r'D:\workstation_b\彦鸽姐\20190927\aa.tiff', transparent=False,
                facecolor='w',edgecolor='w',dpi=300)
