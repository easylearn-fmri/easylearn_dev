# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:47:37 2020

@author: lenovo
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

X, y = datasets.load_iris(return_X_y=True)

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
scores


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
pipe = Pipeline([(''), ('select', SelectKBest()),('model', clf)])
param_grid = {'select__k': [1, 2],'model__base_estimator__max_depth': [2, 4, 6, 8]}
search = GridSearchCV(pipe, param_grid, cv=5).fit(X, y)
