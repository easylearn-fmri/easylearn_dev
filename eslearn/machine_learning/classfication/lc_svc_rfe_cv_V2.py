# -*- coding: utf-8 -*-
"""
Created on Wed self.decision  5 21:12:49 2018
1、对特征进行归一化、主成分降维（可选）后，喂入SVC中进行训练，然后用此model对测试集进行预测
2、采取K-fold的策略

input:
             k=3:k-fold
             step=0.1:rfe step
             num_jobs=1: parallel
             scale_method='StandardScaler':standardization method
             pca_n_component=0.9
             permutation=0
output:
            各种分类效果等
@author: LI Chao
new: return to self
"""
# =============================================================================
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from Utils.lc_featureSelection_rfe import rfeCV
from Utils.lc_dimreduction import pca
import Utils.lc_scaler as scl
#==============================================================================

class SVCRefCv(object):
    """
    利用递归特征消除的方法筛选特征，然后用SVR训练模型，后用cross-validation的方式来验证
    """
    def __init__(self,
                 k=3,
                 seed=10,
                 step=0.1,
                 num_jobs=1,
                 scale_method='StandardScaler',
                 pca_n_component=1,  # 默认不使用pca降维
                 permutation=0,
                 show_results=1,
                 show_roc=0):
        
        """initial parameters"""
        
        self.k=k
        self.seed=seed # 随机种子
        self.step=step
        self.num_jobs=num_jobs
        self.scale_method=scale_method
        self.pca_n_component = pca_n_component
        self.permutation=permutation
        self.show_results=show_results
        self.show_roc=show_roc
        print("SVCRefCv initiated")


    def svc_rfe_cv(self,x,y):
        """Mian function"""
        print('training model and testing using '+ str(self.k)+'-fold CV...\n')
        index_train,index_test=self.fetch_kFold_Index_for_allLabel(x,y,self.k)
        self.predict=pd.DataFrame([])
        self.decision=pd.DataFrame([])
        self.y_real_sorted=pd.DataFrame([])
        self.weight_all=np.zeros([self.k,int((len(np.unique(y))*(len(np.unique(y))-1))/2),x.shape[1]])
        y=np.reshape(y,[-1,])
        for i in range(self.k):
            """split"""
            X_train,y_train=x[index_train[i]],y[index_train[i]]
            X_test,y_test=x[index_test[i]],y[index_test[i]]
            self.y_real_sorted=pd.concat([self.y_real_sorted,pd.DataFrame(y_test)])
            """scale"""
            X_train,X_test=self.scaler(X_train,X_test,self.scale_method)
            """pca"""
            if 0<self.pca_n_component<1:
                X_train,X_test,trained_pca=self.dimReduction(X_train,X_test,self.pca_n_component)
            else:
                pass
            """training"""
            model,weight=self.training(X_train,y_train,\
                 step=self.step, cv=self.k,n_jobs=self.num_jobs,\
                 permutation=self.permutation)
            
            """fetch orignal weight"""
            if 0 < self.pca_n_component< 1:
                weight=trained_pca.inverse_transform(weight)
            self.weight_all[i,:,:]=weight
            
            """test"""
            prd,de=self.testing(model,X_test)
            prd=pd.DataFrame(prd)
            de=pd.DataFrame(de)
            self.predict=pd.concat([self.predict,prd])
            self.decision=pd.concat([self.decision,de])
            
            print('{}/{}\n'.format(i+1,self.k))
         
        """print performances"""
        if self.show_results:
            self.eval_prformance(self.y_real_sorted.values,self.predict.values,self.decision.values)
        return  self
    
    def splitData_kFold_oneLabel(self,x,y):
        """
        random k-fold selection
        """
        kf = KFold(n_splits=self.k,random_state=self.seed)
        sklearn.cross_validation.StratifiedKFold(y, n_folds=self.k, random_state=self.seed)
        
        return X_train, X_test,y_test

    def fetch_kFold_Index_for_allLabel(self,x,y,k):
        """分别从每个label对应的数据中，进行kFole选择，
        然后把某个fold的数据组合成一个大的fold数据"""
        uni_y=np.unique(y)
        loc_uni_y=[np.argwhere(y==uni) for uni in uni_y]

        train_index,test_index=[],[]
        for y_ in loc_uni_y:
            tr_index,te_index=self.fetch_kFold_Index_for_oneLabel(y_,k)
            train_index.append(tr_index)
            test_index.append(te_index)

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

    def fetch_kFold_Index_for_oneLabel(self,originLable,k):
        """获得对某一个类的数据的kfold index"""
        np.random.seed(self.seed)
        kf=KFold(n_splits=k)
        train_index,test_index=[],[]
        for tr_index,te_index in kf.split(originLable):
            train_index.append(originLable[tr_index]), \
            test_index.append(originLable[te_index])       
        return train_index,test_index
    
    def scaler(self,train_X,test_X,scale_method):
        """标准化"""
        train_X,model=scl.scaler(train_X,scale_method)
        test_X=model.transform(test_X)
        return train_X,test_X
    
    def dimReduction(self,train_X,test_X,pca_n_component):
        """降维，如pca"""
        train_X,trained_pca=pca(train_X,pca_n_component)
        test_X=trained_pca.transform(test_X)
        return train_X,test_X,trained_pca
    
    def training(self,x,y,\
                 step, cv,n_jobs,permutation):
        """训练模型"""
        model,weight=rfeCV(x,y,step, cv,n_jobs,permutation)
        return model,weight
    
    def testing(self,model,test_X):
        """用模型预测"""
        predict=model.predict(test_X)
        decision=model.decision_function(test_X)
        return predict,decision
    
    def eval_prformance(self,y_real_sorted,predict,decision):
        """评估模型"""
        # accurcay, self.specificity(recall of negative) and self.sensitivity(recall of positive)        
        self.accuracy= accuracy_score (y_real_sorted,predict)
        report=classification_report(y_real_sorted,predict)
        report=report.split('\n')
        self.specificity=report[2].strip().split(' ')
        self.sensitivity=report[3].strip().split(' ')
        self.specificity=float([spe for spe in self.specificity if spe!=''][2])
        self.sensitivity=float([sen for sen in self.sensitivity if sen!=''][2])
        
        # self.confusion_matrix matrix
        self.confusion_matrix=confusion_matrix(y_real_sorted,predict)

        # roc and self.auc
        if len(np.unique(y_real_sorted))==2:
            fpr, tpr, thresh = roc_curve(y_real_sorted,decision)
            self.auc=roc_auc_score(y_real_sorted,decision)
        else:
            self.auc=None
        
        # print performances
        print('\naccuracy={:.2f}\n'.format(self.accuracy))
        print('sensitivity={:.2f}\n'.format(self.sensitivity))
        print('specificity={:.2f}\n'.format(self.specificity))
        if self.auc:
            print('auc={:.2f}\n'.format(self.auc))
        else:
            print('多分类不能计算auc\n')

        if self.show_roc and self.auc:
            fig,ax=plt.subplots()
            ax.plot(figsize=(5, 5))
            ax.set_title('ROC Curve')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.grid(True)
            ax.plot(fpr, tpr,'-')
            
            """设置坐标轴在axes正中心"""
            ax.spines['top'].set_visible(False) #去掉上边框
            ax.spines['right'].set_visible(False) #去掉右边框
#            ax.spines['bottom'].set_position(('axes',0.5 ))
#            ax.spines['left'].set_position(('axes', 0.5))
        return self
        
if __name__=='__main__':
    from sklearn import datasets
    import lc_svc_rfe_cv_V2 as lsvc
    x,y=datasets.make_classification(n_samples=200, n_classes=2,
                                     n_informative=50,n_redundant=3,
                                     n_features=100,random_state=1)
    sel=lsvc.SVCRefCv(k=3)
    results=sel.svc_rfe_cv(x,y)
    results=results.__dict__
