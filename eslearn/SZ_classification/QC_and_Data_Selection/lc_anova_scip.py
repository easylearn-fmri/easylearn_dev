# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 20:06:03 2019

@author: Chao Li
Email: lichao19870617@gmail.com
"""


from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

import itertools

df2=pd.DataFrame()
df2['group']=list(itertools.repeat(-1.,9))+ list(itertools.repeat(0.,9))+list(itertools.repeat(1.,9))

df2['noise_A']=0.0
for i in data['A'].unique():
    df2.loc[df2['group']==i,'noise_A']=data.loc[data['A']==i,['1','2','3']].values.flatten()
    
df2['noise_B']=0.0
for i in data['B'].unique():
    df2.loc[df2['group']==i,'noise_B']=data.loc[data['B']==i,['1','2','3']].values.flatten()  
    
df2['noise_C']=0.0
for i in data['C'].unique():
    df2.loc[df2['group']==i,'noise_C']=data.loc[data['C']==i,['1','2','3']].values.flatten()  
    
df2