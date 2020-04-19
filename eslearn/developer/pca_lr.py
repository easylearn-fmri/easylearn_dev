#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This demo was copy and revised from sklearn

此是为开发者准备的一个简单的开发demo
假设用户希望
    1. 用主成分分析（PCA）来执行特征降维,并且设置了参数优化范围和参数个数，即max_components = 1,
    min_components = 0.5, number = 5
    2. 用方差分析来筛选特征，并且设置了参数优化范围和参数个数，即max_number = 1(占总特征数的比例),
    min_number = 0.5(占总特征数的比例), number = 5
    3. 用逻辑回归来作为分类器，并且设置了参数优化范围和参数个数，即max_l1_ratio = 1,
    min_l1_ratio = 0, number = 5。注：逻辑回归使用了l1正则。

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import Memory
from shutil import rmtree

from eslearn.model_evaluation.el_evaluation_model_performances import eval_performance

# Datasets
X, y = make_classification(n_features=20, n_redundant=0, n_informative=2)
X_train, y_train = X[:80], y[:80]
X_test, y_test = X[80:], y[80:]

# Memory
location = 'cachedir'
memory = Memory(location=location, verbose=10)

# Make pipeline
pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classify', LogisticRegression(solver='saga', penalty='l1'))
    ], 
    memory=memory
)

#%% Set paramters according to users inputs
# PCA参数
max_components = 0.99,
min_components = 0.5
number_pc = 5
range_dimreduction = np.linspace(min_components, max_components, number_pc).reshape(number_pc,)

# ANOVA参数
max_number = 1,
min_number = 0.5
number_anova = 5
range_feature_selection = np.linspace(min_number, max_number, number_anova).reshape(number_anova,)
# 由于anova检验的特征数必须是整数，所以进行如下的操作，将min/max_number 变为整数
range_feature_selection = np.int16(range_feature_selection * np.shape(X)[1])

# 分类器参数
max_l1_ratio = 1,
min_l1_ratio = 0
number_l1_ratio = 5
range_l1_ratio = np.linspace(min_l1_ratio, max_l1_ratio, number_l1_ratio).reshape(number_l1_ratio,)

# 整体grid search设置
param_grid = [
    {
        'reduce_dim__n_components': range_dimreduction,
        'feature_selection__k': range_feature_selection,
        'classify__l1_ratio': range_l1_ratio,
    },
]

#%% Trains
grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
grid.fit(X_train, y_train)
# 此出可以添加模型持久化的方法，即将grid模型保存到本地，方便用户日后使用该模型

#%% Delete the temporary cache before exiting
memory.clear(warn=False)
rmtree(location)

#%% Prediction
pred = grid.predict(X_test)
dec = grid.predict_proba(X_test)[:,1]

# Evaluate performances
# 非常欢迎您能贡献您的模型评估代码，比如多分类的评估，回归的评估等。
acc, sens, spec, auc = eval_performance(
    y_test, pred, dec, 
    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
    verbose=1, is_showfig=0
)