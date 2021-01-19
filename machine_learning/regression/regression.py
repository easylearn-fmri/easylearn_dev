#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Author: Mengshi Dong <dongmengshi1990@163.com>
"""

import time
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, max_error
from scipy.stats import pearsonr
import pickle

from eslearn.base import BaseMachineLearning, DataLoader
from eslearn.machine_learning.regression._base_regression import BaseRegression
from eslearn.machine_learning.regression._base_regression import StatisticalAnalysis
from eslearn.preprocessing.preprocessing import Denan
from eslearn.model_evaluator import ModelEvaluator


class Regression(BaseMachineLearning, DataLoader, BaseRegression):
    
    def __init__(self, configuration_file, out_dir):
        BaseMachineLearning.__init__(self, configuration_file)
        DataLoader.__init__(self, configuration_file)
        BaseRegression.__init__(self)
        self.metric = mean_absolute_error
        self.out_dir = out_dir
        
    def preprocessing(self):
        # Get all inputs
        self.load_data()
        self.get_all_inputs()
        # Make pipeline
        self.make_sklearn_search_model_(metric=mean_absolute_error)
        
    def main_run(self):
        
        self.preprocessing()
        
        # Get training and test datasets        
        self.target_test_all = []
        self.pred_prob = []
        self.real_score = []
        models = []
        weights = []
        subname = []
        for train_index, test_index in self.method_model_evaluation_.split(self.features_, self.targets_):
            feature_train = self.features_[train_index, :]
            feature_test = self.features_[test_index, :]
            target_train = self.targets_[train_index]
            target_test = self.targets_[test_index]
            
            # Preprocessing
            self.prep_ = Denan(how='median')
            feature_train = self.prep_.fit_transform(feature_train)
            feature_test = self.prep_.transform(feature_test)
            preprocessor.append(self.prep_)
                
            self.target_test_all.extend(target_test)

            subname_ = self.id_[test_index]
            subname.extend(subname_)
            
            # Fit
            self.fit_(self.model_, feature_train, target_train, self.memory)
            models.append(self.model_)
            
            # Get weights
            _, weights_ = self.get_weights_(feature_train, target_train)

            # Predict
            y_prob = self.predict_(self.model_, feature_test)
            
            # Eval performances
            score = self.metric(target_test, y_prob)  
            self.real_score.append(score)
            self.pred_prob.extend(y_prob)

            weights.append(weights_)          
        
        # Eval performances for all fold
        out_name_perf = os.path.join(self.out_dir, "regression_performances.pdf")
        all_score = ModelEvaluator().regression_evaluator(
            self.target_test_all, self.pred_prob, self.real_score, 
            is_showfig=False, is_savefig=True, out_name=out_name_perf
        )

        # Save weight
        self.save_weight(weights, self.out_dir)
        
        # Save outputs
        self.outputs = { 
            "preprocessor": preprocessor, "model":models,
            "subname": subname, "test_targets": self.target_test_all, 
            "test_probability": self.pred_prob, "score": self.real_score 
        }

        pickle.dump(self.outputs, open(os.path.join(self.out_dir, "outputs.pickle"), "wb"))
        return self
    
    def run_statistical_analysis(self):
        # StatisticalAnalysis
        stat = StatisticalAnalysis(self.method_statistical_analysis_,
            self.target_test_all, 
            self.pred_prob,
            self.model_, 
            self.features_, 
            self.targets_, 
            self.prep_, 
            self.param_statistical_analysis_, 
            self.method_model_evaluation_,
            self.real_score,
            self.memory,
            self.out_dir
        )

        stat.fit()

if __name__ == "__main__":
    time_start = time.time()
    reg = Regression(configuration_file=r'F:\一月份线上讲座\regression.json',
                     out_dir=r"F:\一月份线上讲座")
    reg.main_run()
    time_end = time.time()
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)