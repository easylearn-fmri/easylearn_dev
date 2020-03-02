# utf-8
"""
聚类热图
"""

import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import numpy as np

data = pd.read_excel(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Plot\data.xlsx')
data.index = data.iloc[:,0]
data = data.iloc[:,2:]
clname = list(data.columns)
data = data[['q_A',
             'q_A_unmedicated',
             'q_A_medicated',
             'q_B',
             'q_B_unmedicated',
             'q_B_medicated',
             'q_C',
             'q_C_unmedicated',
             'q_C_medicated',]]
#data=data.iloc[0:20:,:]
data.to_csv('D:/data.txt')
 
# 绘制x-y-z的热力图，比如 年-月-销量 的聚类热图 
g = sns.heatmap(data.values, linewidths=None)
g = sns.clustermap(data, figsize=(6,9), cmap='YlGnBu', col_cluster=False, standard_scale = 0)
ax = g.ax_heatmap
label_y = ax.get_yticklabels() 
plt.setp(label_y, fontsize=10, rotation=360, horizontalalignment='left')
label_x = ax.get_xticklabels() 
plt.setp(label_x, fontsize=15, rotation=90)

#设置图片名称，分辨率，并保存
# plt.savefig(r'D:\cluster1.tif', dpi = 600, bbox_inches = 'tight')
plt.show()
