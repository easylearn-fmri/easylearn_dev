#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import time
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from eslearn.base import DataLoader
from eslearn.machine_learning.classfication._base_classificaition import PipelineSearch_

x, y = datasets.make_classification(n_samples=500, n_classes=2,
                                    n_informative=50, n_redundant=3,
                                    n_features=1000, random_state=1)


class Classification(DataLoader, PipelineSearch_):
    
    def __init__(self, configuration_file):
        DataLoader.__init__(self, configuration_file)
        PipelineSearch_.__init__(self)
        self.search_strategy = 'grid'
        self.n_jobs = 2
        self.k = 3

    def classification(self):
        
        # Get all inputs
        self.load_data()
        self.get_all_inputs()

        # Make pipeline
        self.make_pipeline_(
            method_feature_preprocessing=self.method_feature_preprocessing_, 
            param_feature_preprocessing=self.param_feature_preprocessing_, 
            method_dim_reduction=self.method_dim_reduction_, 
            param_dim_reduction=self.param_dim_reduction_, 
            method_feature_selection=self.method_feature_selection_,
            param_feature_selection=self.param_feature_selection_,
            method_machine_learning=self.method_machine_learning_, 
            param_machine_learning=self.param_machine_learning_
        )
        
        # Get training and test datasets        
        cv = self.method_model_evaluation_ 
        target_test_all = []
        for train_index, test_index in cv.split(self.features_, self.targets_):
            feature_train = self.features_[train_index, :]
            feature_test = self.features_[test_index, :]
            target_train = self.targets_[train_index]
            target_test = self.targets_[test_index]
            target_test_all.extend(target_test)

            # Resample
            # ros = RandomOverSampler(random_state=0)
            # feature_train, target_train = ros.fit_resample(feature_train, target_train)
            # print(f"After re-sampling, the sample size are: {sorted(Counter(target_train).items())}")
        
            self.fit_pipeline_(feature_train, target_train)
            self.get_weights_(feature_train, target_train)
            yhat, y_prob = self.predict(feature_test)
            accuracy = accuracy_score(yhat, target_test)
            
        return yhat, y_prob, accuracy


if __name__ == "__main__":
    time_start = time.time()
    clf = Classification(configuration_file=r'D:\My_Codes\easylearn\eslearn\GUI\test\configuration_file.json') 
    clf.classification()
    time_end = time.time()
    print(clf.param_search_)
    # print(clf.pipeline_)
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)