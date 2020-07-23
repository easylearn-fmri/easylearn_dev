# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:45:04 2018
one vs one multi-class classfication
@author: lenovo
"""
from sklearn.svm import SVC
from sklearn import datasets
#
X,y=datasets.make_classification(n_samples=1000, n_features=200, n_informative=2, n_redundant=2,
					n_repeated=0, n_classes=3, n_clusters_per_class=1, weights=None,
 					flip_y=0.01, class_sep=1.0, hypercube=True,shift=0.0, scale=1.0, 
					shuffle=True, random_state=None)
#
def oneVsOne(X,y):
    clf = SVC(decision_function_shape='ovr')
    clf.fit(X, y) 
    predict=clf.predict(X)
    dec=clf.decision_function(X)
    return predict,dec