# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:29:02 2018
sns.set(stynamele='ticks', palette='muted', color_codes=True, font_scale=1.5)
sns.set_stynamele('dark')
主题 stynamele：darkgrid, whitegrid, dark, white, ticks，默认为darkgrid。
sns.set_palette:deep, muted, bright, pastel, dark, colorblind
sns.set_contexnamet('notebook', rc={'lines.linewidth':1.5})
sns.despine()：
对于白底（white，whitegrid）以及带刻度（ticks）而言，顶部的轴是不需要的，默认为去掉顶部的轴；
sns.despine(left=True)：去掉左部的轴，也即 yname 轴；
注意这条语句要放在 plot 的动作之后，才会起作用；
@author: li chao
"""

# 载入绘图模块
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 散点图和拟合直线 以及分布


def scatter_LC(df, x='x', y='y', color='g', marker='o'):
    sns.set(context='paper', style='whitegrid', palette='colorblind', font='sans-serif',font_scale=1, color_codes=False, rc=None)
#    sns.JointGrid(data=df, x=x, y=y).plot(sns.regplot, sns.distplot)
    sns.regplot(data=df, x=x, y=y, fit_reg=1, color=color, marker=marker)
    
    # set
    ax = plt.gca()
    sns.despine()
    xticklabel = ax.get_xticklabels()
    yticklabel = ax.get_yticklabels()
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    
    plt.setp(xticklabel, size=10,rotation=0, horizontalalignment='right')
    plt.setp(yticklabel, size=10,rotation=0, horizontalalignment='right')
    plt.xlabel(xlabel, size=15, rotation=0)
    plt.ylabel(ylabel, size=15, rotation=0)
    # plt.show()


if __name__ == "__main__":
    plt.figure(figsize=(10,8))
    signal_p = r'D:\WorkStation_2018\Workstation_Old\Workstation_2019_Insomnia_caudate_GCA\GCA\Y2X\ROISignals_T2\ROISignals_ROISignal_patients.txt'
    signal_c = r'D:\WorkStation_2018\Workstation_Old\Workstation_2019_Insomnia_caudate_GCA\GCA\Y2X\ROISignals_T2\ROISignals_ROISignal_controls.txt'
    s = r'D:\WorkStation_2018\Workstation_Old\Workstation_2019_Insomnia_caudate_GCA\GCA\Y2X\ROISignals_T2\sas.txt'
    df_signal_p = pd.read_csv(signal_p,header=None)
    df_signal_c = pd.read_csv(signal_c, header=None)
    df_scale = pd.read_csv(s,header=None)
    df = pd.concat([df_signal_p,df_signal_c],axis=0)
    dia = np.hstack([np.zeros(31,), np.ones(47,)])
    df['dia'] = pd.DataFrame(dia)
    
    
    df = pd.concat([df_signal_p,df_scale],axis=1)
    df.columns = ['x','y']
    scatter_LC(df, 'x', 'y', color='#008B8B', marker='o')
    plt.show()
    plt.savefig('pDMN_sas.tif', dpi=600)


