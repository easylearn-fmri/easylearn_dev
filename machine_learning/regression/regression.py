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
import pickle

from eslearn.base import BaseMachineLearning, DataLoader
from eslearn.machine_learning.regression._base_regression import BaseRegression
from eslearn.preprocessing.preprocessing import denan
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
        self.make_pipeline_()
        
    def main_run(self):
        
        self.preprocessing()
        
        # Get training and test datasets        
        self.target_test_all = []
        self.pred_prob = []
        self.real_score = []
        weights = []
        subname = []
        for train_index, test_index in self.method_model_evaluation_.split(self.features_, self.targets_):
            feature_train = self.features_[train_index, :]
            feature_test = self.features_[test_index, :]
            target_train = self.targets_[train_index]
            target_test = self.targets_[test_index]
            
            # Preprocessing
            feature_train, fill_value = denan(feature_train, how='median')
            if np.isnan(feature_test).any().sum() > 0:
                feature_test = pd.DataFrame(feature_test).fillna(fill_value)
                
            self.target_test_all.extend(target_test)

            subname_ = self.id_[test_index]
            subname.extend(subname_)
            
            # Fit
            self.fit_(feature_train, target_train)
            self.get_weights_(feature_train, target_train)

            # Predict
            y_prob = self.predict_(feature_test)
            
            # Eval performances
            score = self.metric(target_test, y_prob)  
            self.real_score.append(score)
            self.pred_prob.extend(y_prob)

            weights.append(self.weights_)          
        

        # Eval performances for all fold
        out_name_perf = os.path.join(self.out_dir, "regression_performances.pdf")
        all_score = ModelEvaluator().regression_evaluator(
            self.target_test_all, self.pred_prob, self.real_score, 
            verbose=1, is_showfig=True, is_savefig=True, out_name=out_name_perf
        )

        # Save weight
        self.save_weight(weights, self.out_dir)

        # Statistical analysis
        print("Statistical analysis...\n")
        self.run_statistical_analysis()
        
        # Save outputs
        outputs = { "subname": subname, "test_targets": self.target_test_all, "test_probability": self.pred_prob, 
                    "score": self.real_score, "pvalue_spec": self.pvalue_score, 
        }

        pickle.dump(outputs, open(os.path.join(self.out_dir, "outputs.pickle"), "wb"))
        return self
    
    def run_statistical_analysis(self):
        type_dict = {"Binomial test":self.binomial_test, "Permutation test":self.permutation_test}
        type_dict[self.method_statistical_analysis_]()
        return self

    # TODO: change it to calculate P value of Pearson's correlation coefficient
    def binomial_test(self):
        k = np.sum(np.array(self.target_test_all) - np.array(self.pred_label)==0)
        n = len(self.target_test_all)
        self.pvalue_acc, sum_prob, prob, randk = el_binomialtest.binomialtest(n, k, 0.5, 0.5)
        self.pvalue_auc = None
        self.pvalue_sens = None
        self.pvalue_spec = None
        print(f"p value for acc = {self.pvalue_acc:.3f}")
        return self

    def permutation_test(self):
        print(f"Permutation test: {self.param_statistical_analysis_} times...\n")
        
        self.preprocessing()
        
        self.permuted_score = []

        for i in range(self.param_statistical_analysis_):
            print(f"{i+1}/{self.param_statistical_analysis_}...\n")      
            permuted_score = []
            for train_index, test_index in self.method_model_evaluation_.split(self.features_, self.targets_):
                feature_train = self.features_[train_index, :]
                feature_test = self.features_[test_index, :]
                
                # Preprocessing
                feature_train, fill_value = denan(feature_train, how='median')
                if np.isnan(feature_test).any().sum() > 0:
                    feature_test = pd.DataFrame(feature_test).fillna(fill_value)
                
                permuted_target_train = self.targets_[train_index][np.random.permutation(len(train_index))]
                target_train = self.targets_[train_index]
                target_test = self.targets_[test_index]

                # Fit
                self.fit_(feature_train, permuted_target_train)

                # Predict
                y_prob = self.predict_(feature_test)
                
                # Eval performances
                score = self.metric(target_test, y_prob)  
                permuted_score.append(score)
             
            # Average performances of one permutation
            self.permuted_score.append(np.mean(permuted_score))

        # Get p values
        self.pvalue_score = self.calc_pvalue(self.permuted_score, np.mean(self.real_score))
        print(f"p value for score = {self.pvalue_score:.3f}")
        return self

    @staticmethod
    def calc_pvalue(permuted_performance, real_performance):
        return (np.sum(np.array(permuted_performance) <= np.array(real_performance)) + 1) / (len(permuted_performance) + 1)
    

if __name__ == "__main__":
    time_start = time.time()
    clf = Regression(configuration_file=r'F:\一月份线上讲座\regression.json',
                     out_dir=r"F:\一月份线上讲座")
    clf.main_run()
    time_end = time.time()
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)