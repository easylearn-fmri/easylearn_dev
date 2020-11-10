# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:13:55 2020

@author: lenovo
"""

import pickle
import numpy as np

rs = pickle.load(open(r"F:\耿海洋workshop\demo_data\outputs.pickle", "rb"))


X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)


from sklearn.preprocessing import MinMaxScaler
X = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
scaler = MinMaxScaler()
print(scaler.fit(X))

print(scaler.data_max_)

print(scaler.transform(data))




print(scaler.transform([[2, 2]]))
