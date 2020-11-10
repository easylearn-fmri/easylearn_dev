# -*- coding: utf-8 -*-
"""
箱图
当我们的数据是num_subj*num_var，且有几个诊断组时，我们一般希望把var name作为x，把var value作为y，把诊断组作为hue
来做箱图，以便于观察每个var的组间差异。
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
class BoxPlot():
    # initial parameters
    def __init__(self,
                       data_path=r'D:\others\彦鸽姐\final_data.xlsx',
                       x_location=np.arange(5,13,1),
                       x_name='脑区',
                       y_name='reho',
                       hue_name='分组',
                       hue_order=[2,1],
                       if_save_figure=0,
                       figure_name='violin.tiff'):
           #======================================
           self.data_path=data_path
           self.x_location=x_location
           self.x_name=x_name
           self.y_name=y_name
           self.hue_name=hue_name
           self.hue_order=hue_order
           self.if_save_figure=if_save_figure
           self.figure_name=figure_name
           
#====================================================   
    def data_preparation(self):

        # load data
        df = pd.read_excel(self.data_path,index=False)
        
        # 筛选数据
        df_selected=df.iloc[:,self.x_location]
        
        # 将'[]'去除
        df_selected = pd.DataFrame(df_selected, dtype=np.str) 
        df_selected=df_selected.mask(df_selected =='[]', None, inplace=False)
        df_selected=df_selected.dropna()
        col_to_float=list(set(list(df_selected.columns))-set([self.hue_name]))
        df_selected[col_to_float] = pd.DataFrame(df_selected[col_to_float], dtype=np.float32) 
#        a=pd.Series(df_selected['HAMD']).str.contains('\d',regex=False)

        
        
        #把需要呈现的数据concat到一列
        n_subj,n_col=df_selected.shape
        df_long=pd.DataFrame([])
        for nc in range(n_col):
            df_long=pd.concat([df_long,df_selected.iloc[:,nc]])
            
        # 整理columns
        col_name=list(df_selected.columns)
        col_name_long=[pd.DataFrame([name]*n_subj) for name in col_name]
        col_name_long=pd.concat(col_name_long)
        
        #整理分组标签
        group=pd.DataFrame([])
        for i in range(n_col):
            group=pd.concat([group,df[self.hue_name].loc[df_selected.index]])
            
        #整合
        col_name_long.index=df_long.index # 解决index不统一问题
        self.data=pd.concat([group,col_name_long,df_long],axis=1)
        
        # 加列名
        self.data.columns=[self.hue_name, self.x_name, self.y_name]
        
        return self
        
#=========================================================================    
    def plot(self):

        # plot
        self.f,self.ax= plt.subplots()
        
        # box框架
        self.data=self.data_preparation().data
        self.ax=sns.boxplot(x=self.x_name, 
                       y=self.y_name,
                       hue=self.hue_name,
                       order=None,
                       hue_order=self.hue_order,
                       data=self.data,
                       
                       palette="Set1",
                       saturation=0.7,
                       width=0.5,
                       fliersize=2,
                       whis=None,
                       notch=False,
                       dodge=True)     
                
        #设置网格
#        plt.grid(axis="y",ls='--',c='g')
        
        # 设置label，以及方位
        label_x = self.ax.get_xticklabels()
        label_y = self.ax.get_yticklabels()
        plt.setp(label_x, size=15,rotation=0, horizontalalignment='right')
        plt.setp(label_y, size=15,rotation=0, horizontalalignment='right')

        # save figure
        if self.if_save_figure:
            self.f.savefig(self.figure_name, dpi=300, bbox_inches='tight')
      
        return self

if __name__ == '__main__':
    sel = BoxPlot(data_path=r'D:\WorkStation_2018\WorkStation_dynamicFC_V3\Data\results_cluster\results_of_individual\temploral_properties.xlsx',
                     x_location=np.arange(1, 3, 1),
                     hue_name='group',
                     hue_order=[1, 3, 2, 4])
    
    df =sel.data_preparation()
    sel.plot()
    plt.savefig('MDT.tif',dpi=600)


