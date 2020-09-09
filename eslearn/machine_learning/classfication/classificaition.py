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
        x_train, x_test, y_train, y_test = train_test_split(self.features_, self.targets_, test_size=0.30, random_state=42)
        
        cv = self.method_model_evaluation_
        for train_index, test_index in cv.split(self.data, self.label):
            data_train = self.data[train_index, :]
            data_test = self.data[test_index, :]
            label_train = self.label[train_index]
            label_test = self.label[test_index]
            label_test_all.extend(label_test)

            # Resample
            ros = RandomOverSampler(random_state=0)
            data_train, label_train = ros.fit_resample(data_train, label_train)

            print(f"After re-sampling, the sample size are: {sorted(Counter(label_train).items())}")
            
        
        self.fit_pipeline_(x_train, y_train)
        self.get_weights_(x_train, y_train)
        yhat, y_prob = self.predict(x_test)
        accuracy = accuracy_score(yhat, y_test)
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