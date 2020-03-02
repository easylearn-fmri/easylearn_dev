# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 19:23:42 2018
ElasticNetCV
Minimizes the objective function:
    1 / (2 * n_samples) * ||y_train - Xw||^2_2
    + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

l1_ratio = 1 is the lasso penalty

a * L1 + b * L2
where:
    alpha = a + b and l1_ratio = a / (a + b)
@author: lenovo
"""
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
import numpy as np

class ElasticNetCV():
    
    def __init__(sel,
                 k=10,
                 l1_ratio=np.linspace(0.1,1,10),
                 alphas=np.linspace(0.001,100,100)):
        
        sel.k=k
        sel.l1_ratio=l1_ratio
        sel.alpha=alphas
    
    def train(sel,x_train,y_train):
        
        sel.regr = ElasticNetCV(random_state=0,cv=sel.k,
                                l1_ratio=sel.l1_ratio,
                                alphas=sel.alpha)
        
        sel.regr.fit(x_train,y_train)
        
        sel.best_alpha=sel.regr.alpha_
        sel.best_l1_ratio=sel.regr.l1_ratio_
        sel.best_coef=sel.regr.coef_
        sel.best_intercept=sel.regr.intercept_
        
        return sel
        
    def test(sel,x_test):
        sel.pred=sel.regr.predict(x_test)
        return sel


if __name__=='__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    x, y = make_regression(n_samples=500,n_features=5, random_state=0)
    x_train, x_test, y_train, y_test = \
                            train_test_split(x, y, random_state=0)
                            
    import lc_elasticNetCV as ENCV
    sel=ENCV.ElasticNetCV()
    sel.train(x_train,y_train)
    results=sel.test(x_test).__dict__
    
    r,p=pearsonr(results['pred'],y_test)
    print('r={}\np={}'.format(r,p))
    