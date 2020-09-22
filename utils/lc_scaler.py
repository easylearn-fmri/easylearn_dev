# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 14:33:00 2018

@author: lenovo
"""
# import
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# def


def scaler(X, method):
    if method == 'StandardScaler':
        model = StandardScaler()
        stdsc_x = model.fit_transform(X)
        return stdsc_x, model
    
    elif method == 'MinMaxScaler':
        model = MinMaxScaler()
        mima_x = model.fit_transform(X)
        return mima_x, model
#    origin_data = model.inverse_transform(mm_data)
    else:
        print(f'Please specify the standardization method!')
        return
    


def scaler_apply(train_x, test_x, scale_method):
    """
    Apply model to test data
    """
    train_x, model = scaler(train_x, scale_method)
    test_x = model.transform(test_x)
    return train_x, test_x
