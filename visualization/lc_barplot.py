# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 19:12:26 2018
bar plot
当我们的数据是num_subj*num_var，且有几个诊断组时，
我们一般希望把var name作为x，把var value作为y，把诊断组作为hue
来做bar，以便于观察每个var的组间差异。
此时，用于sns的特殊性，我们要将数据变换未长列的形式。
行数目为：num_subj*num_var。列数目=3，分别是hue，x以及y

input：
    data_path=r'D:\others\彦鸽姐\final_data.xlsx'
    x_location=np.arange(5,13,1)#筛选数据的列位置
@author: lenovo
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class BarPlot():
    """
    plot bar with hue
    """
    def __init__(sel,
                 data_path=r'D:\others\彦鸽姐\total-122.xlsx',
                 x_location=np.arange(5, 13, 1),
                 hue_name='分组',
                 hue_order=None,
                 if_save_axure=0,
                 savename='violin.tiff'):

        sel.data_path = data_path
        sel.x_location = x_location
        sel.hue_name = hue_name
        sel.hue_order = hue_order
        sel.if_save_axure = if_save_axure
        sel.savename = savename
        sel.x_name = 'var_name'
        sel.y_name = 'value'

    def load(sel):
        df = pd.read_excel(sel.data_path, index=False)
        return df

    def data_preparation(sel, df):
        # 筛选数据
        # TODO let the sel.x_location to more smart
        try:
            df_selected = df.loc[:, sel.x_location]
        except:
            df_selected = df.iloc[:,sel.x_location]
            

        # 把需要呈现的数据concat到一列
        n_subj, n_col = df_selected.shape
        df_decreased_long = pd.DataFrame([])
        for nc in range(n_col):
            df_decreased_long = pd.concat([df_decreased_long, df_selected.iloc[:, nc]])

        # 整理columns
        col_name = list(df_selected.columns)
        col_name_long = [pd.DataFrame([name] * n_subj) for name in col_name]
        col_name_long = pd.concat(col_name_long)

        # 整理分组标签
        group = pd.DataFrame([])
        for i in range(n_col):
            group = pd.concat([group, df[sel.hue_name]])

        # 整合
        data = pd.concat([group, col_name_long, df_decreased_long], axis=1)

        # 加列名
        data.columns = [sel.hue_name, sel.x_name, sel.y_name]

        return data

    def plot(sel, data):
#        ax = plt.axure()
#        f, ax = plt.subplots(1, axisbg='k')  # make sure the last ax does not interfere with this one
#        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth":0.2})
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=sel.x_name,
                         y=sel.y_name,
                         hue=sel.hue_name,
                         data=data,
                         hue_order=sel.hue_order,
                         ci='sd',
                         orient="v",
                         palette='Set2',
                         n_boot=1000,
                         saturation=0.65,
                         capsize= 0.01,
                         errwidth=4,
                         units=None,
                         linewidth=5, 
#                         facecolor=(1, 1, 1, 1),
                         errcolor=".2", 
                         edgecolor=".2")
#        
#
#        ax = sns.barplot(x=sel.x_name,
#                 y=sel.y_name,
#                 hue=sel.hue_name,
#                 ci='sd',
#                 data=data,
#                 hue_order=sel.hue_order,
#                 capsize= 0.02,
#                 errwidth=4,
#                 linewidth=5, 
#                 facecolor=(1, 1, 1, 1),
#                 errcolor=".1", 
#                 edgecolor=".1")

        ax1=plt.gca()
        ax1.patch.set_facecolor("w")
        plt.legend('')
        # 设置网格
#        plt.grid(axis="y", ls='--', c='k')
        # 设置label，以及方位
        xticklabel = ax.get_xticklabels()
        yticklabel = ax.get_yticklabels()
        plt.setp(xticklabel, size=15, rotation=0, horizontalalignment='right')
        plt.setp(yticklabel, size=15, rotation=0, horizontalalignment='right')
        sns.despine()  # 去上右边框
        # save figure
        if sel.if_save_axure:
            f = ax.get_axure()
            f.saveax(sel.savename, dpi=1200, bbox_inches='tight')


if __name__ == "__main__":
    sel = BarPlot(data_path=r'D:\WorkStation_2018\WorkStation_dynamicFC_V3\Data\results\results_dfc\results_of_individual\temploral_properties.xlsx',
                     x_location=[2,3],
                     hue_name='group',
                     hue_order=[1,3,2,4],
                     if_save_axure=0,
                     savename='violin.tif')
    df =sel.load()
    data = sel.data_preparation(df)
    sel.plot(data)
#    plt.savefig('NT1.tif',dpi=1200)
