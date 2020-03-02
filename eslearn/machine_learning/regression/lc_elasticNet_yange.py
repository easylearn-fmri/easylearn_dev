# -*- coding: utf-8 -*-
"""
@author: lI Chao
"""

import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\utils')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\classfication')
from lc_read_write_Mat import read_mat,write_mat
import lc_elasticNetCV as ENCV
        
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class classify_using_FC():
    
    def __init__(sel):
        sel.file_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\DynamicFC_length17_step1_screened'#mat文件所在路径
        sel.dataset_name=None # mat文件打开后的名字
        sel.scale=r'D:\WorkStation_2018\WorkStation_dynamicFC\Scales\8.30大表.xlsx'
        sel.save_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic'
        sel.feature='mean' #用均数还是std等('mean'/'std'/'staticFC')
        
        sel.mask=np.ones([114,114]) #特征矩阵的mask
        sel.mask=np.triu(sel.mask,1)==1 # 只提取上三角（因为其他的为重复）
        
        sel.n_processess=10
        sel.if_save_post_mat=1 #保存后处理后的mat？
        
        sel.random_state=2
        
    
    def postprocessing_features(sel,mat):
        # 准备特征：比如取上三角，拉直等
         return mat[sel.mask]
        

    def machine_learning(sel,order=[3,4]):
        # elasticNet
        print('elasticNetCV')
        sel=ENCV.elasticNetCV()
        sel.train(x_train,y_train)
        sel.test(x_test)
         
        results=sel.test(x_test).__dict__

# =============================================================================
#         
#         # rfe
#         import lc_svc_rfe_cv_V2 as lsvc
#         model=lsvc.svc_rfe_cv(k=5,pca_n_component=0.85)
#         
#         results=model.main_svc_rfe_cv(x.values,y)
# =============================================================================
        
        results=results.__dict__
    
        
        
        return results


if __name__=='__main__':
    import lc_classify_FC as Clasf
    sel=Clasf.classify_using_FC()
    
    results=sel.load_allmat()
    results=sel.gen_label()
    result=sel.machine_learning(order=[1,3])