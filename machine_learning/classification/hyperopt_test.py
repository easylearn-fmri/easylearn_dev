# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:03:43 2020

@author: lenovo
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from hpsklearn import HyperoptEstimator, svc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


X, y = make_classification(n_features=4, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


estim = HyperoptEstimator(classifier=svc('mySVC'))

estim = LinearSVC()

estim.fit(X_train, y_train)

print(estim.score(X_test, y_test))


#%%
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from sklearn.datasets import load_iris
from hyperopt import tpe
import numpy as np

# Download the data and split into training and test sets

iris = load_iris()

X = iris.data
y = iris.target

test_size = int(0.2 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]

# Instantiate a HyperoptEstimator with the search space and number of evaluations

estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                          preprocessing=any_preprocessing('my_pre'),
                          algo=tpe.suggest,
                          max_evals=100,
                          trial_timeout=120)

# Search the hyperparameter space based on the data

estim.fit(X_train, y_train)

# Show the results

print(estim.score(X_test, y_test))
# 1.0

print(estim.best_model())
