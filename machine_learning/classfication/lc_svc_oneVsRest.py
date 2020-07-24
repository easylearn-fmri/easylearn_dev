# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:56:39 2018
multi-class classfication using one vs rest
In practice, one-vs-rest classification is usually preferred, 
since the results are mostly similar, 
but the runtime is significantly less.
@author: lenovo
"""
from sklearn import datasets
from sklearn.svm import LinearSVC
import numpy as np
#
X,y=datasets.make_classification(n_samples=1000, n_features=200, n_informative=2, n_redundant=2,
					n_repeated=0, n_classes=3, n_clusters_per_class=1, weights=None,
 					flip_y=0.01, class_sep=1.0, hypercube=True,shift=0.0, scale=1.0, 
					shuffle=True, random_state=None)
#
def oneVsRest(X,y):
    lin_clf = LinearSVC()
    lin_clf.fit(X, y) 
    predict=lin_clf.predict(X)
    dec = lin_clf.decision_function(X)
    return predict,dec

if __name__=='__main__':
    predict,dec=oneVsRest(X,y)