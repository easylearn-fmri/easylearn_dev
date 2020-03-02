# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:55:51 2018

@author: lenovo
"""
import sys
sys.path.append(r'D:\myCodes\MVPA_LIChao\MVPA_Python\MVPA\utils')
from lc_read_write_Mat import read_mat
import numpy as np
import pandas as pd


hcPath=r'I:\dynamicALFF\Results\DALFF\50_0.9\Statistical_Results\Signal\ROISignals_ROISignal_FWHM4_HC.mat'
szPath=r'I:\dynamicALFF\Results\DALFF\50_0.9\Statistical_Results\Signal\ROISignals_ROISignal_FWHM4_SZ.mat'
bdPath=r'I:\dynamicALFF\Results\DALFF\50_0.9\Statistical_Results\Signal\ROISignals_ROISignal_FWHM4_BD.mat'
mddPath=r'I:\dynamicALFF\Results\DALFF\50_0.9\Statistical_Results\Signal\ROISignals_ROISignal_FWHM4_MDD.mat'

dataset_struct,datasetHC=read_mat(hcPath,'ROISignals')
dataset_struct,datasetSZ=read_mat(szPath,'ROISignals')
dataset_struct,datasetBD=read_mat(bdPath,'ROISignals')
dataset_struct,datasetMDD=read_mat(mddPath,'ROISignals')

meanHC=pd.DataFrame(np.mean(datasetHC,axis=0))
meanSZ=pd.DataFrame(np.mean(datasetSZ,axis=0))
meanBD=pd.DataFrame(np.mean(datasetBD,axis=0))
meanMDD=pd.DataFrame(np.mean(datasetMDD,axis=0))

allData=pd.concat([meanHC,meanSZ,meanBD,meanMDD],axis=1)
allData.index=['左侧额中回/额上回 ','右侧额上回（靠内)','右侧前扣带回 ','右侧尾状核','左侧尾状核',
                         '右侧putamen','左侧putamen','右侧前岛叶',
                         '左侧前岛叶','右侧杏仁核 ','左侧杏仁核 ',
                         '右侧海马','左侧海马','右侧海马旁回','左侧海马旁回','右侧舌回','左侧舌回',
                         '右侧cuneus','左侧cuneus','右侧angular gyrus','右侧中央后回']

allData.columns=['HC','SZ','BD','MDD']




