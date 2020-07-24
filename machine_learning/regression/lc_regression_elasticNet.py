# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:15:54 2018
1.以功能连接/动态功能连接矩阵为特征，来进行回归
2.本程序使用的算法为svc（交叉验证）
3.当特征是动态连接时，使用标准差或者均数等来作为特征。也可以自己定义
4.input：
    所有人的.mat FC/dFC
5.output:
    机器学习的相应结果，以字典形式保存再result中。
 
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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
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

        
    def load_allmat(sel):
        # 多线程
        s=time.time()
        print('loading all mat...\n')
         
        # 判断是否有FC mat文件
        if os.path.exists(os.path.join(sel.save_path,sel.feature+'.mat')):
            sel.mat=pd.DataFrame(read_mat(os.path.join(sel.save_path,sel.feature+'.mat'),None))
            print('Already have {}\nloaded all mat!\nrunning time={:.2f}'.format(sel.feature+'.mat',time.time()-s))
        else:
            
            sel.all_mat=os.listdir(sel.file_path)
            all_mat_path=[os.path.join(sel.file_path,all_mat_) for all_mat_ in sel.all_mat]
            
            cores = multiprocessing.cpu_count()
            if sel.n_processess>cores:
                sel.n_processess=cores-1
                    
            len_all=len(all_mat_path)
            sel.mat=pd.DataFrame([])
            
            # 特征用std还是mean
            if sel.feature=='mean':
                ith=1
            elif sel.feature=='std':
                ith=0
            elif sel.feature=='staticFC':
                ith=0
            else:
                print('###还未添加其他衡量dFC的指标,默认使用std###\n')
                ith=0
            
            # load mat...
            with ThreadPoolExecutor(sel.n_processess) as executor:
                for i, all_mat_ in enumerate(all_mat_path):
                            task=executor.submit(sel.load_onemat_and_processing, i,all_mat_,len_all,s) 
                            sel.mat=pd.concat([sel.mat,pd.DataFrame(task.result()[ith]).T],axis=0)
                            
            # 保存后处理后的mat文件
            if sel.if_save_post_mat:
                write_mat(fileName=os.path.join(sel.save_path,sel.feature+'.mat'),
                          dataset_name=sel.feature,
                          dataset=np.mat(sel.mat.values))
                print('saved all {} mat!\n'.format(sel.feature))
 
                
        
    def load_onemat_and_processing(sel,i,all_mat_,len_all,s):
        # load mat
        mat=read_mat(all_mat_,sel.dataset_name)
        
        # 计算方差，均数等。可扩展。(如果时静态FC，则不执行)
        if sel.feature=='staticFC':
            mat_std,mat_mean=mat,[]
        else:
            mat_std,mat_mean=sel.calc_std(mat)
        
        # 后处理特征，可扩展
        if sel.feature=='staticFC':
            mat_std_1d,mat_mean_1d=sel.postprocessing_features(mat_std),[]
        else:
            mat_std_1d=sel.postprocessing_features(mat_std)
            mat_mean_1d=sel.postprocessing_features(mat_mean)
        
        # 打印load进度
        if i%10==0 or i==0:
            print('{}/{}\n'.format(i,len_all))
        
        if i%50==0 and i!=0:
            e=time.time()
            remaining_running_time=(e-s)*(len_all-i)/i
            print('\nremaining time={:.2f} seconds \n'.format(remaining_running_time))
        
        return mat_std_1d,mat_mean_1d
    
    def calc_std(sel,mat):
        mat_std=np.std(mat,axis=2)
        mat_mean=np.mean(mat,axis=2)
        return mat_std,mat_mean
    
    def postprocessing_features(sel,mat):
        # 准备特征：比如取上三角，拉直等
         return mat[sel.mask]
        
    
    def gen_label(sel):
        
        # 判断是否已经存在label
        if os.path.exists(os.path.join(sel.save_path,'folder_label.xlsx')):
            sel.label=pd.read_excel(os.path.join(sel.save_path,'folder_label.xlsx'))['诊断']
            print('\nAlready have {}\n'.format('folder_label.xlsx'))
       
        else:
            # identify label for each subj
            id_subj=pd.Series(sel.all_mat).str.extract('([1-9]\d*)')
            
            scale=pd.read_excel(sel.scale)
            
            id_subj=pd.DataFrame(id_subj,dtype=type(scale['folder'][0]))
                     
            sel.label=pd.merge(scale,id_subj,left_on='folder',right_on=0,how='inner')['诊断']
            sel.folder=pd.merge(scale,id_subj,left_on='folder',right_on=0,how='inner')['folder']
            
            # save folder and label
            if sel.if_save_post_mat:
                sel.label_folder=pd.concat([sel.folder,sel.label],axis=1)
                sel.label_folder.to_excel(os.path.join(sel.save_path,'folder_label.xlsx'),index=False)
               
        return sel

    def machine_learning(sel,order=[3,4]):
        
        # label
        y=pd.concat([sel.label[sel.label.values==order[0]] , sel.label[sel.label.values==order[1]]])
        y=y.values
        
        # x/sel.mat
        if os.path.exists(os.path.join(sel.save_path,sel.feature+'.mat')):
            sel.mat=pd.DataFrame(read_mat(os.path.join(sel.save_path,sel.feature+'.mat'),None))
        
        x=pd.concat([sel.mat.iloc[sel.label.values==order[0],:] , sel.mat.iloc[sel.label.values==order[1],:]])
        
#        #平衡测试
#        y=np.hstack([y,y[-1:-70:-1]])
#        x=pd.concat([x,x.iloc[-1:-70:-1]],axis=0)
        
#        y=y[60:]
#        x=x.iloc[60:,:]
#        print(sum(y==0),sum(y==1))
        
        # 置换y
#        rand_ind=np.random.permutation(len(y))
#        y=y[rand_ind]

        # cross-validation
        # 1) split data to training and testing datasets
        x_train, x_test, y_train, y_test = \
                            train_test_split(x, y, random_state=sel.random_state)
                            
                            

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