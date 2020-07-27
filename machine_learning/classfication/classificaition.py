#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import time
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from eslearn.base import BaseMachineLearning
from eslearn.machine_learning.classfication._base_classificaition import BaseClassification, PipelineSearch_

x, y = datasets.make_classification(n_samples=200, n_classes=2,
                                    n_informative=50, n_redundant=3,
                                    n_features=100, random_state=1)


# x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42)

class Classification(BaseMachineLearning, PipelineSearch_):
    
    def __init__(self):
        super(BaseMachineLearning, self).__init__()
        super(PipelineSearch_, self).__init__()
        self.search_strategy = 'grid'
        self.n_jobs = 2

    def classification(self, 
                       x=None, 
                       y=None,
                       method_feature_preprocessing=None, 
                       param_feature_preprocessing=None,
                       method_dim_reduction=None,
                       param_dim_reduction=None,
                       method_feature_selection=None,
                       param_feature_selection=None,
                       method_machine_learning=None,
                       param_machine_learning=None,
    ):
        
        
        
        self.make_pipeline_(
            method_feature_preprocessing=method_feature_preprocessing, 
            param_feature_preprocessing=param_feature_preprocessing, 
            method_dim_reduction=method_dim_reduction, 
            param_dim_reduction=param_dim_reduction, 
            method_feature_selection=method_feature_selection,
            param_feature_selection=param_feature_selection,
            method_machine_learning=method_machine_learning, 
            param_machine_learning=param_machine_learning
        )

        self.fit_pipeline_(x_train, y_train)
        self.get_weights_(x_train, y_train)
        yhat, y_prob = self.predict(x_test)
        accuracy = accuracy_score(yhat, y_test)
        return yhat, y_prob, accuracy


if __name__ == "__main__":
    time_start = time.time()
    clf = Classification()
    clf.get_configuration_(configuration_file=r'F:\Python378\Lib\site-packages\eslearn\GUI\test\configuration_file.json')
    clf.get_preprocessing_parameters()
    clf.get_dimension_reduction_parameters()
    clf.get_feature_selection_parameters()
    clf.get_unbalance_treatment_parameters()
    clf.get_machine_learning_parameters()
    clf.get_model_evaluation_parameters()
    
    method_feature_preprocessing = clf.method_feature_preprocessing
    param_feature_preprocessing= clf.param_feature_preprocessing

    method_dim_reduction = clf.method_dim_reduction
    param_dim_reduction = clf.param_dim_reduction

    method_feature_selection = clf.method_feature_selection
    param_feature_selection = clf.param_feature_selection

    method_machine_learning = clf.method_machine_learning
    param_machine_learning = clf.param_machine_learning
    
    yhat, y_prob, accuracy = clf.classification(
        method_feature_preprocessing=method_feature_preprocessing, 
        param_feature_preprocessing=param_feature_preprocessing,
        method_dim_reduction=method_dim_reduction,
        param_dim_reduction=param_dim_reduction,
        method_feature_selection=method_feature_selection,
        param_feature_selection=param_feature_selection, 
        method_machine_learning=method_machine_learning, 
        param_machine_learning=param_machine_learning,
        x=x, 
        y=y
    )
    
    
    
    time_end = time.time()
    print(clf.param_search_)
    print(clf.pipeline_)
    print(f"accuracy = {accuracy}")
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)