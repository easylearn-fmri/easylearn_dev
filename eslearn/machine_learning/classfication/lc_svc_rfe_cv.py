# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:52:26 2018
rfe-svm-CV
input:
             k=3:k-fold
             step=0.1:rfe step
             num_jobs=1: parallel
             scale_method='StandardScaler':standardization method
             pca_n_component=0.9
             permutation=0
@author: lenovo
"""
import sys  
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\Machine_learning (Python)\Machine_learning\classfication')
from lc_featureSelection_rfe import rfeCV
import lc_dimreduction as dimreduction
import lc_scaler as scl

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


class svc_rfe_cv():
     # initial parameters
    def __init__(self,
                 k=3,
                 step=0.1,
                 num_jobs=1,
                 scale_method='StandardScaler',
                 pca_n_component=0.8,
                 permutation=0,
                 show_results=1,
                 show_roc=0):
        self.k=k
        self.step=step
        self.num_jobs=num_jobs
        self.scale_method=scale_method
        self.pca_n_component=pca_n_component
        self.permutation=permutation
        self.show_results=show_results
        self.show_roc=show_roc
    #
    def main_svc_rfe_cv(self,x,y):
        #
        print('\n'+'#'*10+' Running... '+'#'*10+'\n')
        index_train,index_test=self.fetch_kFold_Index_for_allLabel(x,y,self.k)
        predict=pd.DataFrame([])
        dec=pd.DataFrame([])
        y_real_sorted=pd.DataFrame([])
        y=np.reshape(y,[len(y),])
        for i in range(self.k):
            print('{}/{}\n'.format(i+1,self.k))
            # split
            x_train,y_train=x[index_train[i]],y[index_train[i]]
            X_test,y_test=x[index_test[i]],y[index_test[i]]
            y_real_sorted=pd.concat([y_real_sorted,pd.DataFrame(y_test)])
            # scale
            x_train,X_test=self.scaler(x_train,X_test,self.scale_method)
            # pca
            x_train,X_test,trained_pca=self.dimReduction(x_train,X_test,self.pca_n_component)
            # train
            model,weight=self.training(x_train,y_train,\
                 step=self.step, cv=self.k,n_jobs=self.num_jobs,\
                 permutation=self.permutation)
            # fetch orignal weight
            weight=trained_pca.inverse_transform(weight)
            # test
            prd,de=self.testing(model,X_test)
            prd=pd.DataFrame(prd)
            de=pd.DataFrame(de)
            predict=pd.concat([predict,prd])
            dec=pd.concat([dec,de])
         
        # 打印并显示模型性能
        if self.show_results:
            self.evalPrformance(dec,predict,y_real_sorted)
            
        return  predict,dec,y_real_sorted,weight
    
    def splitData_kFold_oneLabel(self,x,y,k):
        kf = KFold(n_splits=k)
        x_train, X_test=[],[]
        y_test=[]
        for train_index, test_index in kf.split(x):
            x_train.append(x[train_index]), X_test.append( x[test_index])
            y_test.append(y[test_index])
        return x_train, X_test,y_test
            
        
    def fetch_kFold_Index_for_allLabel(self,x,y,k):
        #分别从每个label对应的数据中，进行kFole选择，
        #然后把某个fold的数据组合成一个大的fold数据
        uni_y=np.unique(y)
        loc_uni_y=[np.argwhere(y==uni) for uni in uni_y]
        #
        train_index,test_index=[],[]
        for y_ in loc_uni_y:
            tr_index,te_index=self.fetch_kFold_Index_for_oneLabel(y_,k)
            train_index.append(tr_index)
            test_index.append(te_index)
        #
        indexTr_fold=[]
        indexTe_fold=[]
        for k_ in range(k):
            indTr_fold=np.array([])
            indTe_fold=np.array([])
            for y_ in range(len(uni_y)):
                indTr_fold=np.append(indTr_fold,train_index[y_][k_])
                indTe_fold=np.append(indTe_fold,test_index[y_][k_])
            indexTr_fold.append(indTr_fold)
            indexTe_fold.append(indTe_fold)
        index_train,index_test=[],[]
        for I in indexTr_fold:
            index_train.append([int(i) for i in I ])
        for I in indexTe_fold:
            index_test.append([int(i) for i in I])
        return index_train,index_test
    #
    def fetch_kFold_Index_for_oneLabel(self,originLable,k):
        kf=KFold(n_splits=k)
        train_index,test_index=[],[]
        for tr_index,te_index in kf.split(originLable):
            train_index.append(originLable[tr_index]), \
            test_index.append(originLable[te_index])       
        return train_index,test_index
    
    def scaler(self,train_X,test_X,scale_method):
        train_X,model=scl.scaler(train_X,scale_method)
        test_X=model.transform(test_X)
        return train_X,test_X
    
    def dimReduction(self,train_X,test_X,pca_n_component):
        train_X,trained_pca=dimreduction.pca(train_X,pca_n_component)
        test_X=trained_pca.transform(test_X)
        return train_X,test_X,trained_pca
    
    def training(self,x,y,\
                 step, cv,n_jobs,permutation):
    #    refCV
        model,weight=rfeCV(x,y,step, cv,n_jobs,permutation)
        return model,weight
    
    def testing(self,model,test_X):
        predict=model.predict(test_X)
        dec=model.decision_function(test_X)
        return predict,dec
    
    def evalPrformance(self,dec,predict,y_real_sorted):
        
        # accurcay, specificity(recall of negative) and sensitivity(recall of positive)        
        accuracy= accuracy_score (y_real_sorted.values,predict.values)
        report=classification_report(y_real_sorted.values,predict.values)
        report=report.split('\n')
        specificity=report[2].strip().split(' ')
        sensitivity=report[3].strip().split(' ')
        specificity=float([spe for spe in specificity if spe!=''][2])
        sensitivity=float([sen for sen in sensitivity if sen!=''][2])
        
        # confusion matrix
        confusion=confusion_matrix(y_real_sorted.values,predict.values)

        # roc and auc
        fpr, tpr, thresh = roc_curve(y_real_sorted.values,dec.values)
        auc=roc_auc_score(y_real_sorted.values,dec.values)
        
        # print performances
#        print('混淆矩阵为:\n{}'.format(confusion))
        
        print('\naccuracy={:.2f}\n'.format(accuracy))
        print('sensitivity={:.2f}\n'.format(sensitivity))
        print('specificity={:.2f}\n'.format(specificity))
        print('AUC={:.2f}\n'.format(auc))

        if self.show_roc:
            plt.figure(figsize=(5, 5))
            plt.title('ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid(True)
            plt.plot(fpr, tpr,'-')
    #        plt.savefig('roc.png')
           
        return accuracy,sensitivity,specificity,auc,confusion
        

#        
if __name__=='__main__':
    from sklearn import datasets
    import lc_svc_rfe_cv as lsvc
    x,y=datasets.make_classification(n_samples=500, n_features=500,random_state=1)
    sel=lsvc.svc_rfe_cv(k=5)
    predict,dec,y_real_sorted,weight=sel.main_svc_rfe_cv(x,y)