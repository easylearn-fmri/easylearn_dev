# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 18:27:38 2018
ElasticNet
Minimizes the objective function:
    1 / (2 * n_samples) * ||y_train - Xw||^2_2+ alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

l1_ratio = 1 is the lasso penalty

a * L1 + b * L2
where:
    alpha = a + b and l1_ratio = a / (a + b)
@author: lenovo
"""
# =============================================================================
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn import preprocessing
import numpy as np

class ElasticNet():
    
    def __init__(sel,
                 x_train=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\x_train342.npy',
                 y_train=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\y_train342.npy',

                 x_val=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\x_test38.npy',
                 y_val=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\y_test38.npy',
                    
                 x_test=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\x_test206.npy',
                 y_test=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\y_test206.npy',
                                  
                 l1_ratio=0.5,
                 alpha=0.1):
                
        sel.l1_ratio=l1_ratio
        sel.alpha=alpha
        
        sel.x_train=x_train
        sel.y_train=y_train
        
        sel.x_val=x_val
        sel.y_val=y_val
        
        sel.x_test=x_test
        sel.y_test=y_test

    def load_data_and_label(sel,x,y,label_col):
        x=np.load(x)
        y=np.load(y)[:,label_col]
        return x,y

    def normalization(sel,data):
        # because of our normalization level is on subject, 
        # we should transpose the data matrix on python(but not on matlab)
        scaler = preprocessing.StandardScaler().fit(data.T)
        z_data=scaler.transform(data.T) .T
        return z_data
    
    def train(sel,x_train,y_train):
        sel.regr = ElasticNet(random_state=0,l1_ratio=sel.l1_ratio,alpha=sel.alpha)
        sel.regr.fit(x_train,y_train)
        
        sel.coef=sel.regr.coef_
        sel.intersept=sel.regr.intercept_ 
        
        return sel
        
    def test(sel,x_test):
        sel.pred=sel.regr.predict(x_test)
        return sel


if __name__=='__main__':
    x_train,y_train=[[0,0], [1, 1], [2, 2]], [0, 1, 2]
    import lc_elasticNet as EN
    sel=EN.ElasticNet()
    sel.train(x_train,y_train)
    sel.test(x_train)
    