# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:35:09 2018
plot hot map
@author: li chao
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')

import statsmodels.stats.multitest as mlt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Utils.lc_read_write_Mat import read_mat
##=====================================================================

# 生成数据
x=pd.read_excel(r'D:\workstation_b\彦鸽姐\20190927\r_DTI.xlsx',header=None,index=None)
p=pd.read_excel(r'D:\workstation_b\彦鸽姐\20190927\p_DTI.xlsx',header=None,index=None)

mask=pd.read_excel(r'D:\workstation_b\彦鸽姐\20190927\p_DTI.xlsx',header=None,index=None)
mask = mask > 0.05

data=pd.read_excel(r'D:\workstation_b\彦鸽姐\20190927\DTI.xlsx')
netsize = np.shape(p)

#results = mlt.multipletests(np.reshape(p.values,np.size(p)), alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
#mask=np.reshape(results[0],netsize)

#mask=mask==False
header=list(data.columns)
header=header[3:]

# plot
f, (ax) = plt.subplots(figsize=(20,20))
sns.heatmap(x,
            ax=ax,
            annot=True,
            annot_kws={'size':9,'weight':'normal', 'color':'k'},fmt='.3f',
            cmap='RdBu_r',
#            center=0,
            square=True,
            linewidths = 0.005, 
            linecolor= 'k',
            mask=mask,
            vmin=-1,
            vmax=1)

#ax.set_title('hot map')
ax.set_xlabel('')
ax.set_ylabel('')

ax.set_xticklabels(header)
ax.set_yticklabels(header)
# 设置选中，以及方位
label_x = ax.get_xticklabels()
label_y = ax.get_yticklabels()

# 
plt.subplots_adjust(top = 1, bottom = 0.5, right = 1, left = 0.5, hspace = 0, wspace = 0)
#plt.margins(0,0)

plt.setp(label_x, rotation=90,horizontalalignment='right')
plt.setp(label_y, rotation=0,horizontalalignment='right')
plt.setp(label_x, fontsize=15)
plt.setp(label_y, fontsize=15)
plt.show()

#ax.imshow(x)

plt.savefig(r'D:\workstation_b\彦鸽姐\20190927\r_dti.tiff',
            transparent=True, dpi=600, pad_inches = 0)