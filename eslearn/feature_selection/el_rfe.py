# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:06:48 2018
Recursive feature elimination (RFE) and RFE-Cross-validation(nested)
@author: Li Chao
"""

from sklearn.feature_selection import RFE
#from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import numpy as np


def rfe(x, y, step=0.1, n_features_to_select=10,
        permutation=0):
    """sigle rfe"""
    estimator = SVC(kernel="linear")
    selector = RFE(estimator, step=step,
                   n_features_to_select=n_features_to_select)

    selector.fit(x, y)
#    mask=selector.support_
    rank = selector.ranking_
    return rank

def rfeCV(x, y, step, cv, n_jobs,
          permutation=0):
    """equal to nested rfe"""
    n_samples, n_features = x.shape
    estimator = SVC(kernel="linear")  # TOO: Add other classifiers
    selector = RFECV(estimator, step=step, cv=cv, n_jobs=n_jobs)
    selector = selector.fit(x, y)
    mask = selector.support_
#    rank=selector.ranking_
    optmized_model = selector.estimator_
    w = optmized_model.coef_  # 当为多分类时，w是2维向量
    weight = np.zeros([w.shape[0], n_features])
    weight[:, mask] = w
#    selector.score(x, y)
#    y_pred=selector.predict(x)
#    r=np.corrcoef(y,y_pred)[0,1]
    return selector, weight


##
if __name__ == '__main__':
    """example"""
    from sklearn import datasets
    x, y = datasets.make_classification(n_samples=200, n_classes=2,
                                        n_informative=50, n_redundant=3,
                                        n_features=100, random_state=1)

    selector, weight = rfeCV(x, y, step=0.1, cv=3, n_jobs=1,
                             permutation=0)

    y_pred = selector.predict(x)
