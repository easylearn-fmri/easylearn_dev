# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:03:39 2018
小提琴图
当我们的数据是num_subj*num_var，且有几个诊断组时，我们一般希望把var name作为x，把var value作为y，把诊断组作为hue
来做小提琴图，以便于观察每个var的组间差异。
此时，用于sns的特殊性，我们要将数据变换未长列的形式。
行数目为：num_subj*num_var。列数目=3，分别是hue，x以及y

input：
    data_path=r'D:\others\彦鸽姐\final_data.xlsx'
    x_location=np.arange(5,13,1)#筛选数据的列位置
    
未来改进：封装为类，增加可移植性
@author: lenovo
"""
#==========================================================
# 载入绘图模块
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#==========================================================
class ViolinPlot():
    # initial parameters
    def __init__(sel,
                       data_path=r'D:\others\彦鸽姐\final_data.xlsx',
                       x_location=np.arange(5,13,1),
                       x_name='脑区',
                       y_name='reho',
                       hue_name='分组',
                       hue_order=[2,1],
                       if_save_figure=0,
                       figure_name='violin.tiff'):
           #======================================
           sel.data_path=data_path
           sel.x_location=x_location
           sel.x_name=x_name
           sel.y_name=y_name
           sel.hue_name=hue_name
           sel.hue_order=hue_order
           sel.if_save_figure=if_save_figure
           sel.figure_name=figure_name
           
#====================================================   
    def data_preparation(sel):

        # load data
        df = pd.read_excel(sel.data_path,index=False)
        
        # 筛选数据
        df_selected=df.iloc[:,sel.x_location]
        
        #把需要呈现的数据concat到一列
        n_subj,n_col=df_selected.shape
        df_decreased_long=pd.DataFrame([])
        for nc in range(n_col):
            df_decreased_long=pd.concat([df_decreased_long,df_selected.iloc[:,nc]])
            
        # 整理columns
        col_name=list(df_selected.columns)
        col_name_long=[pd.DataFrame([name]*n_subj) for name in col_name]
        col_name_long=pd.concat(col_name_long)
        
        #整理分组标签
        group=pd.DataFrame([])
        for i in range(n_col):
            group=pd.concat([group,df[sel.hue_name]])
            
        #整合
        sel.data=pd.concat([group,col_name_long,df_decreased_long],axis=1)
        
        # 加列名
        sel.data.columns=[sel.hue_name, sel.x_name, sel.y_name]
        
        return sel
         
    def plot(sel):
        sel.data=sel.data_preparation().data
        # plot
        plt.plot(figsize=(5, 15))
        
        # 小提琴框架
        ax=sns.violinplot(x=sel.x_name, y=sel.y_name,hue=sel.hue_name,
                    data=sel.data,palette="Set2",
                    split=False,scale_hue=True,hue_order=sel.hue_order,
                    orient="v",inner="box")
#        
           
        
        # 设置label，以及方位
        label_x = ax.get_xticklabels()
        label_y = ax.get_yticklabels()
        plt.setp(label_x, size=10,rotation=0, horizontalalignment='right')
        plt.setp(label_y, size=10,rotation=0, horizontalalignment='right')
        
        # save figure
        if sel.if_save_figure:
            f.savefig(sel.figure_name, dpi=300, bbox_inches='tight')
            
#        plt.hold
#        #加点/风格1
#        sns.swarmplot(x=sel.x_name, y=sel.y_name,hue=sel.hue_name,data=sel.data, 
#                              color="w", alpha=.5,palette="Set1")
##        
        #加点/风格2
#        sns.stripplot(x=sel.x_name, y=sel.y_name,hue=sel.hue_name,data=sel.data, 
#                              color="w", alpha=.5,palette="Set1", jitter=False)
        
#        plt.show()
          
        return sel

if __name__ == '__main__':
    sel = ViolinPlot(data_path=r'D:\WorkStation_2018\WorkStation_dynamicFC_V3\Data\results_cluster\results_of_individual\temploral_properties.xlsx',
                     x_location=np.arange(1, 2, 1),
                     hue_name='group',
                     hue_order=None)
    df =sel.data_preparation()
    sel.plot()


