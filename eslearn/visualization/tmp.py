# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:39:06 2019

@author: lenovo
"""

from scipy import io
x = io.loadmat(r'D:\WorkStation_2018\WorkStation_dimensionPLS\Data\ROISignals\ROICorrelation_FisherZ_ROISignal_00003.mat')
x = x['ROICorrelation_FisherZ']

f, (ax) = plt.subplots(figsize=(20,20))
sns.heatmap(x,
            ax=ax,
            annot=False,
            annot_kws={'size':9,'weight':'normal', 'color':'k'},fmt='.3f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths = False, 
            linecolor= [0.6,0.6,0.6],
            mask=None,
            vmin=-1,
            vmax=1)

#plt.subplots_adjust(top = 1, bottom = 0.5, right = 1, left = 0.5, hspace = 0, wspace = 0)

#plt.savefig(r'D:\workstation_b\彦鸽姐\20190927\aa.tiff',
#            transparent=True, dpi=300, pad_inches = 0)