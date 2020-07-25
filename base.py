#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Base class for all modules
"""

import json
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso, BayesianRidge

class BaseMachineLearning:

    def __init__(self):
        pass

    def argparse_(self, configuration_file):
        """Parse the configuration file
        """

        with open(configuration_file, 'r', encoding='utf-8') as config:
                    configuration = config.read()
        self.configuration = json.loads(configuration)

        return self

    def load_data(self):
        pass

    def get_preprocessing_parameters(self):
        self.method_feature_preprocessing = None
        self.param_feature_preprocessing = []
        
        preprocessing = self.configuration.get('feature_engineering', {}).get('feature_preprocessing', None)
        if preprocessing:
            self.method_feature_preprocessing = (list(preprocessing.keys())[0] if list(preprocessing.keys())[0] != 'None' else None)
    
            for key in preprocessing.keys():
                for key_ in preprocessing.get(key).keys():
                    if key_ != []:
                        for key__ in preprocessing.get(key).get(key_).keys():
                             self.param_feature_preprocessing.append(eval(preprocessing.get(key).get(key_).get(key__)))
             
            self.param_feature_preprocessing = (tuple(self.param_feature_preprocessing) if self.param_feature_preprocessing != [] else None)
        
        return self

    def get_dimension_reduction_parameters(self):
        self.method_dimension_reduction = None
        self.param_dimension_reduction = {}
        
        
        dimension_reduction = self.configuration.get('feature_engineering', {}).get('dimreduction', None)
        if dimension_reduction:
            self.method_dimension_reduction = (list(dimension_reduction.keys())[0] if list(dimension_reduction.keys())[0] != 'None' else None)
    
            for key in dimension_reduction.keys():
                for key_ in dimension_reduction.get(key).keys():
                    if key_ != []:
                        for key__ in dimension_reduction.get(key).get(key_).keys():
                             self.param_dimension_reduction.update({"dim_reduction__"+key_:eval(dimension_reduction.get(key).get(key_).get(key__))})
             
        return self
        

    def get_feature_selection_parameters(self):
        self.method_feature_selection = None
        self.param_feature_selection = {}
        
        
        feature_selection = self.configuration.get('feature_engineering', {}).get('feature_selection', None)
        if feature_selection:
            self.method_feature_selection = (list(feature_selection.keys())[0] if list(feature_selection.keys())[0] != 'None' else None)
    
            for key in feature_selection.keys():
                for key_ in feature_selection.get(key).keys():
                    if key_ != []:
                        for key__ in feature_selection.get(key).get(key_).keys():
                             self.param_feature_selection.update({"feature_selection__"+key_:eval(feature_selection.get(key).get(key_).get(key__))})
             
        return self

    def get_unbalance_treatment_parameters(self):
        self.method_unbalance_treatment = None
        self.param_unbalance_treatment = {}
        
        
        unbalance_treatment = self.configuration.get('feature_engineering', {}).get('unbalance_treatment', None)
        if unbalance_treatment:
            self.method_unbalance_treatment = (list(unbalance_treatment.keys())[0] if list(unbalance_treatment.keys())[0] != 'None' else None)
    
            for key in unbalance_treatment.keys():
                for key_ in unbalance_treatment.get(key).keys():
                    if key_ != []:
                        for key__ in unbalance_treatment.get(key).get(key_).keys():
                             self.param_unbalance_treatment.update({"unbalance_treatment__"+key_:eval(unbalance_treatment.get(key).get(key_).get(key__))})
             
        return self

    def get_machine_learning_parameters(self):
         self.configuration.get('machine_learning', None)

    def get_model_evaluation_parameters(self):
        self.configuration.get('model_evaluation', None)

    def get_statistical_analysis_parameters(self):
        self.configuration.get('statistical_analysis', None)

    def get_visualization_parameters(self):
        self.configuration.get('visualization', None)

if __name__ == '__main__':
    base = BaseMachineLearning()
    base.argparse_(configuration_file=r'F:\Python378\Lib\site-packages\eslearn\GUI\test\configuration_file.json')
    base.get_preprocessing_parameters()
    base.get_dimension_reduction_parameters()
    base.get_feature_selection_parameters()
    base.get_unbalance_treatment_parameters()
    

    print(base.method_feature_preprocessing)
    print(base.param_feature_preprocessing)
    
    print(base.method_dimension_reduction)
    print(base.param_dimension_reduction)
    
    print(base.method_feature_selection)
    print(base.param_feature_selection)
    
    print(base.method_unbalance_treatment)
    print(base.param_unbalance_treatment)
    
    