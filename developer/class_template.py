# -*- coding: utf-8 -*-
"""This module is used to preprocess data.

Created on Wed Jul  4 13:57:15 2018
@author: Li Chao
Email:lichao19870617@gmail.com
GitHub account name: lichao312214129
Institution (company): Brain Function Research Section, The First Affiliated Hospital of China Medical University, Shenyang, Liaoning, PR China. 
License: MIT
"""


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


class Preprocessing():
    '''This class is used to preprocess features

    TODO: add other preprocessing methods

    Method 1: preprocess data in group level, namely one feature(column) by one feature(column).
    Method 2: preprocess data in subject level, namely one subject(row) by one subject(row).

    Parameters:
    ----------
        data_preprocess_method: string
            how to preprocess the features, 'StandardScaler' or 'MinMaxScaler'
        data_preprocess_level: string
            which level to preprocess features, 'subject' or 'group'
    Attibutes:
    ----------
        None
    '''

    def __init__(self, data_preprocess_method='StandardScaler', data_preprocess_level='subject'):
        self.data_preprocess_method = data_preprocess_method
        self.data_preprocess_level = data_preprocess_level

    def data_preprocess(self, feature_train, feature_test):
        '''This function is used to preprocess features
       
        Method 1: preprocess data in group level, namely one feature(column) by one feature(column).
        Method 2: preprocess data in subject level, namely one subject(row) by one subject(row).

        Parameters
        ----------
            feature_train: numpy.ndarray
                features in training dataset
            feature_test: numpy.ndarray
                features in test dataset
        Returns
        ------
            preprocessed training features and test features.
        '''

        # Method 1: Group level preprocessing.
        if self.data_preprocess_level == 'group':
            feature_train, model = self.scaler(feature_train, self.data_preprocess_method)
            feature_test = model.transform(feature_test)
        elif self.data_preprocess_level == 'subject':
            # Method 2: Subject level preprocessing.
            scaler = preprocessing.StandardScaler().fit(feature_train.T)
            feature_train = scaler.transform(feature_train.T) .T
            scaler = preprocessing.StandardScaler().fit(feature_test.T)
            feature_test = scaler.transform(feature_test.T) .T
        else:
            print('Please provide which level to preprocess features\n')
            return

        return feature_train, feature_test

    def scaler(self, X, method):
        """The low level method
        """

        if method == 'StandardScaler':
            model = StandardScaler()
            stdsc_x = model.fit_transform(X)
            return stdsc_x, model
        
        elif method == 'MinMaxScaler':
            model = MinMaxScaler()
            mima_x = model.fit_transform(X)
            return mima_x, model
        else:
            print(f'Please specify the standardization method!')
            return
        
    def scaler_apply(self, train_x, test_x, scale_method):
        """Apply model to test data
        """

        train_x, model = self.scaler(train_x, scale_method)
        test_x = model.transform(test_x)
        return train_x, test_x

