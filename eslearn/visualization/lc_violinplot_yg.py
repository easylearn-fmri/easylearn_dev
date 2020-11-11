# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:21:27 2018
小提琴图:把升高的脑区和降低的脑区分开做小提琴图
增加的脑区：
    x_location=np.arange(13,17,1)
减低的脑区
    x_location=np.arange(5,13,1)
@author: lenovo
"""

import sys
sys.path.append(r'D:\myCodes\MVPA_LIChao\MVPA_Python\plot')
import lc_violinplot as violinplot
import lc_barplot as barplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


data1 = r'D:\WorkStation_2018\Workstation_Old\Workstation_2019_Insomnia_caudate_GCA\Y2X\ROISignal_OFC_controls\ROISignals_ROISignal_controls.txt'
data2 = r'D:\WorkStation_2018\Workstation_Old\Workstation_2019_Insomnia_caudate_GCA\Y2X\ROISignal_OFC_patients\ROISignals_ROISignal_patients.txt'
data1 = pd.read_csv(data1, header=None)
data2 = pd.read_csv(data2, header=None)
df = pd.concat([data1, data2], axis=0)
df.index = np.arange(0,78)
df['group'] = pd.DataFrame(np.hstack([np.zeros(47,), np.ones(31,)]))

plt.plot(figsize=(4,9))
ax = sns.barplot(x='group',
             y=0,
             data=df,
             orient="v")

ax1=plt.gca()
ax1.patch.set_facecolor("w")
# 设置网格
#        plt.grid(axis="y", ls='--', c='k')
# 设置label，以及方位
xticklabel = ax.get_xticklabels()
yticklabel = ax.get_yticklabels()
plt.setp(xticklabel, size=7, rotation=45, horizontalalignment='right')
plt.setp(yticklabel, size=10, rotation=0, horizontalalignment='right')
sns.despine()  # 去上右边框
plt.savefig('bar.tif', dpi=600)