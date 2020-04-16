# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:02:33 2020

@author: lenovo
"""

# AdaBoost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)
base_clf = LogisticRegression(C=1.)
clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=100)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()


# # Ridge classification
# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import RidgeClassifier
# X, y = load_breast_cancer(return_X_y=True)
# clf = RidgeClassifier().fit(X, y)
# clf.score(X, y)


# LogisticRegression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
X, y = make_classification(n_classes=3, n_informative=5, n_redundant=0, random_state=42)
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y)
