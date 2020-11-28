#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Author: Mengshi Dong <dongmengshi1990@163.com>
"""

import time
import os
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, max_error

from eslearn.base import BaseMachineLearning, DataLoader
from eslearn.machine_learning.regression._base_regression import BaseRegression
from eslearn.model_evaluator import ModelEvaluator


x, y = datasets.make_regression(n_samples=200, n_informative=50, n_features=100, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


class Regression(BaseMachineLearning, DataLoader, BaseRegression):
    
    def __init__(self, configuration_file, out_dir):
        BaseMachineLearning.__init__(self, configuration_file)
        DataLoader.__init__(self, configuration_file)
        BaseRegression.__init__(self)
        self.metric = mean_absolute_error
        self.out_dir = out_dir

    def main_run(self):
        
        # Get all inputs
        self.load_data()
        self.get_all_inputs()

        # Make pipeline
        self.make_pipeline_()
        
       # Get training and test datasets        
        cv = self.method_model_evaluation_ 
        target_test_all = []
        pred_prob = []
        self.real_score = []
        weights = []
        for train_index, test_index in cv.split(self.features_, self.targets_):
            feature_train = self.features_[train_index, :]
            feature_test = self.features_[test_index, :]
            target_train = self.targets_[train_index]
            target_test = self.targets_[test_index]
            target_test_all.extend(target_test)

            # Resample
            imbalance_resample = self.method_unbalance_treatment_
            feature_train, target_train = imbalance_resample.fit_resample(feature_train, target_train)
            print(f"After re-sampling, the sample size are: {sorted(Counter(target_train).items())}")
            
            # Fit
            self.fit_(feature_train, target_train)
            self.get_weights_(feature_train, target_train)
            y_prob = self.predict_(feature_test)
            
            # Eval performances
            score = self.metric(target_test, y_prob)  

            self.real_score.append(score)
            pred_prob.extend(y_prob)
            weights.append(self.weights_)          
        
        # Save weight
        self.save_weight(weights, self.out_dir)

        return score
        

if __name__ == "__main__":
    time_start = time.time()
    clf = Regression(configuration_file=r'D:\My_Codes\easylearn\eslearn\GUI\test\configuration_file_reg.json',
                     out_dir=r"D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\GUI\tests")
    clf.main_run()
    time_end = time.time()
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)