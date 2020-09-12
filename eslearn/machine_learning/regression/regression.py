#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Author: Mengshi Dong <dongmengshi1990@163.com>
"""

import time
import os
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, max_error

from eslearn.base import DataLoader
from eslearn.machine_learning.regression._base_regression import PipelineSearch_


x, y = datasets.make_regression(n_samples=200, n_informative=50, n_features=100, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


class Regression(DataLoader, PipelineSearch_):
    
    def __init__(self, configuration_file):
        DataLoader.__init__(self, configuration_file)
        PipelineSearch_.__init__(self, location=os.path.dirname(configuration_file))
        self.search_strategy = 'grid'
        self.n_jobs = -1
        self.metric = mean_absolute_error

    def regression(self):
        
        # Get all inputs
        self.load_data()
        self.get_all_inputs()

        # Make pipeline
        self.make_pipeline_()
        
        self.fit_pipeline_(x_train, y_train)
        self.get_weights_(x_train, y_train)
        yhat = self.predict(x_test)
        score = self.metric(yhat, y_test)

        print(f"score = {score}")
        return yhat, score
        

if __name__ == "__main__":
    time_start = time.time()
    clf = Regression(configuration_file=r'D:\My_Codes\easylearn\eslearn\GUI\test\configuration_file_reg.json') 
    clf.regression()
    time_end = time.time()
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)