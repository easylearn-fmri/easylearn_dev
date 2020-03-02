# -*- coding: utf-8 -*-
"""
For 胜男姐
数据输入格式为excel
svc
@author: Li Chao
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import os
import numpy as np
import pandas as pd
# import pickle
from sklearn.preprocessing import LabelEncoder


class SVCForDataFromExcel(object):
    def __init__(sel,
                #==============================================================
                # 等号之间为所有输入
                patients_path,  # excel数据位置
                col_name_of_label,  # label所在列的项目名字
                col_num_of_data,  # 特征所在列的序号（第哪几列）
                is_save_results,  # 是否保存结果（True：保存；False：不保存）
                save_path,  # 结果保存在哪个地方

                # Default parameters
                kfold=10,  # 几折交叉验证，此处为5折交叉验证
                scale_method='MinMaxScaler',
                pca_n_component=0.8,  # 主成分降维，不降维设为1  # label项目名字
                show_results=1,  # 是否在屏幕打印结果
                show_roc=0,  # 是否在屏幕显示ROC曲线
                _seed=100,
                step=0.05
                #==============================================================
                ):
        sel.patients_path = patients_path
        sel.col_name_of_label = col_name_of_label
        sel.col_num_of_data = col_num_of_data
        sel.is_save_results = is_save_results
        sel.save_path = save_path
        sel.kfold = kfold
        sel.scale_method = scale_method
        sel.pca_n_component = pca_n_component
        sel.show_results = show_results
        sel.show_roc = show_roc
        sel._seed = _seed
        sel.step = step
        print("SVCForDataFromExcel initated!")

    def _load_data(sel):
        # According suffix to load data
        # TODO: expanding to other data type
        if os.path.basename(sel.patients_path).split('.')[-1] == 'xlsx' or \
           os.path.basename(sel.patients_path).split('.')[-1] == 'xls':
            data = pd.read_excel(sel.patients_path)
        elif os.path.basename(sel.patients_path).split('.')[-1] == 'csv':
            data = pd.read_csv(sel.patients_path, encoding='gbk',engine='c')
        else:
            print(f'Unspported data type!')
            return
        return data 

    def _prep_data(sel, data):
        """
        Preprocessing data
        """
        data = data.dropna()
#        sel.data_all = sel.data_all.iloc[:-30,:]
        label = data[sel.col_name_of_label].values
        data = data.iloc[:, sel.col_num_of_data].values
        
        # up-resample
    
        le = LabelEncoder()
        le.fit(label)
        label = le.transform(label)
        
        return data, label

    def tr_te(sel, data, label):
        """
        Training and test
        """
        import lc_svc_rfe_cv_V3 as lsvc
        svc = lsvc.SVCRfeCv(pca_n_component=sel.pca_n_component,
                          show_results=sel.show_results,
                          show_roc=sel.show_roc,
                          outer_k=sel.kfold,
                          scale_method=sel.scale_method,
                          step=sel.step,
                          _seed=sel._seed)
        
        results = svc.svc_rfe_cv(data, label)
        return results

    def save(sel):
        import time
        now = time.strftime("%Y%m%d%H%M%S", time.localtime())
        with open(os.path.join(sel.save_path, "".join(["results_", now, "_.pkl"])), "wb") as file:
            pickle.dump(sel.results, file, True)

#        # load pkl file
#        with open("".join(["results_",now,"_.pkl"]),"rb") as file:
#            results = pickle.load(file)

    def main(sel):
        data = sel._load_data()
        data, label = sel._prep_data(data)
        results = sel.tr_te(data, label)
        if sel.is_save_results:
            sel.save()
        return results


if __name__ == "__main__":
    sel = SVCForDataFromExcel(
                patients_path=r'D:\workstation_b\宝宝\allROIvenous_information.csv',  # excel数据位置
                col_name_of_label=r"Lable(malignant-1,benign-0)",  # label所在列的项目名字
                col_num_of_data=np.arange(4, 400),  # 特征所在列的序号（第哪几列）
                is_save_results=0,  # 是否保存结果（True：保存；False：不保存）
                save_path=r"D:\workstation_b\宝宝" # 结果保存在哪个地方
                )
    results = sel.main()
    a=results.__dict__
    for keys in a:
    	print(f'{keys}:{a[keys]}')
#    
    
#    d0 = a[a['Lable(malignant-1,benign-0)']==0]
#    d1 = a[a['Lable(malignant-1,benign-0)']==1]
#    import matplotlib.pyplot as plt
#    plt.hist(d0.iloc[:,20])
#    plt.hist(d1.iloc[:,20])
#    plt.show()
