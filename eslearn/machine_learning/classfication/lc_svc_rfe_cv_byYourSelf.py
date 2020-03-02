# -*- coding: utf-8 -*-
"""
Created on Wed self.decision  5 21:12:49 2018
自定义训练集和测试集，进行训练和测试
rfe-svm-CV
input:
             k=3:k-fold
             step=0.1:rfe step
             num_jobs=1: parallel
             scale_method='StandardScaler':standardization method
             pca_n_component=0.9
             permutation=0
@author: LI Chao
new: 函数统一返回给self
"""
# =============================================================================
import sys  
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\machine_learning_python\Utils')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\classfication')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\machine_learning_python\Machine_learning\neural_network')

from lc_featureSelection_rfe import rfeCV
import lc_pca as pca
import lc_scaler as scl

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#==============================================================================

class svc_rfe_cv():
     # initial parameters
    def __init__(self,
                 k=3,
                 seed=10,
                 step=0.1,
                 num_jobs=1,
                 scale_method='StandardScaler',
                 pca_n_component=0.8,
                 permutation=0,
                 show_results=1,
                 show_roc=0):
        self.k=k
        self.seed=seed # 随机种子
        self.step=step
        self.num_jobs=num_jobs
        self.scale_method=scale_method
        self.pca_n_component=pca_n_component
        self.permutation=permutation
        self.show_results=show_results
        self.show_roc=show_roc


    def main_svc_rfe_cv(self,x,y):
        
        # 自定义训练集和测试集
        print('training model and testing using '+ str(self.k)+'-fold CV...\n')
        
        # split data
        x_train, x_test,y_train,self.y_test=train_test_split(x, y, random_state=0)
            

        # scale
        x_train,x_test=self.scaler(x_train,x_test,self.scale_method)
        
        # pca
        if 0<self.pca_n_component<1:
            x_train,x_test,trained_pca=self.dimReduction(x_train,x_test,self.pca_n_component)
        else:
            pass
        
        # train
        model,weight=self.training(x_train,y_train,\
             step=self.step, cv=self.k,n_jobs=self.num_jobs,\
             permutation=self.permutation)
        
        # fetch orignal weight
        self.weight_all=pd.DataFrame([])
        if 0<self.pca_n_component<1:
            weight=trained_pca.inverse_transform(weight)
        self.weight_all=pd.concat([self.weight_all,pd.DataFrame(weight)],axis=1)
        
        # test
        self.predict=pd.DataFrame([])
        self.decision=pd.DataFrame([])
        
        prd,de=self.testing(model,x_test)
        prd=pd.DataFrame(prd)
        de=pd.DataFrame(de)
        self.predict=pd.concat([self.predict,prd])
        self.decision=pd.concat([self.decision,de])
         
        # 打印并显示模型性能
        if self.show_results:
            self.eval_prformance()
            
        return  self
    
    def scaler(self,train_X,test_X,scale_method):
        train_X,model=scl.scaler(train_X,scale_method)
        test_X=model.transform(test_X)
        return train_X,test_X
    
    def dimReduction(self,train_X,test_X,pca_n_component):
        train_X,trained_pca=pca.pca(train_X,pca_n_component)
        test_X=trained_pca.transform(test_X)
        return train_X,test_X,trained_pca
    
    def training(self,x,y,\
                 step, cv,n_jobs,permutation):
    #    refCV
        model,weight=rfeCV(x,y,step, cv,n_jobs,permutation)
        return model,weight
    
    def testing(self,model,test_X):
        predict=model.predict(test_X)
        decision=model.decision_function(test_X)
        return predict,decision
    
    def eval_prformance(self):
        # 此函数返回self
        
        # accurcay, self.specificity(recall of negative) and self.sensitivity(recall of positive)        
        self.accuracy= accuracy_score (self.y_test,self.predict.values)
        report=classification_report(self.y_test,self.predict.values)
        report=report.split('\n')
        self.specificity=report[2].strip().split(' ')
        self.sensitivity=report[3].strip().split(' ')
        self.specificity=float([spe for spe in self.specificity if spe!=''][2])
        self.sensitivity=float([sen for sen in self.sensitivity if sen!=''][2])
        
        # self.confusion_matrix matrix
        self.confusion_matrix=confusion_matrix(self.y_test,self.predict.values)

        # roc and self.auc
        fpr, tpr, thresh = roc_curve(self.y_test,self.decision.values)
        self.auc=roc_auc_score(self.y_test,self.decision.values)
        
        # print performances
#        print('混淆矩阵为:\n{}'.format(self.confusion_matrix))
        
        print('\naccuracy={:.2f}\n'.format(self.accuracy))
        print('sensitivity={:.2f}\n'.format(self.sensitivity))
        print('specificity={:.2f}\n'.format(self.specificity))
        print('auc={:.2f}\n'.format(self.auc))

        if self.show_roc:
            fig,ax=plt.subplots()
            ax.plot(figsize=(5, 5))
            ax.set_title('ROC Curve')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.grid(True)
            ax.plot(fpr, tpr,'-')
            
            #设置坐标轴在axes正中心
            ax.spines['top'].set_visible(False) #去掉上边框
            ax.spines['right'].set_visible(False) #去掉右边框
#            ax.spines['bottom'].set_position(('axes',0.5 ))
#            ax.spines['left'].set_position(('axes', 0.5))
           
        return self
        

#        
if __name__=='__main__':
    
    # 导入1channel，让数据一致，以便于比较
    import lc_CNN_DynamicFC_1channels as CNN
    sel=CNN.CNN_FC_1channels()
    sel=sel.load_data_and_label()
    sel=sel.prepare_data()
    x=sel.data
    y=sel.label
    y=np.argmax(y,axis=1)
    
    import lc_svc_rfe_cv_byYourSelf as lsvc
    model=lsvc.svc_rfe_cv(k=3)
    results=model.main_svc_rfe_cv(x,y)
    results=results.__dict__
