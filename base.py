#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Base class for all modules
"""

import json


class BaseMachineLearning:

    def __init__(self):
        pass

    def argparse_(self, configuration_file):
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
        self.configuration.get('feature_engineering', {}).get('feature_selection', None)

    def get_unbalance_treatment_parameters(self):
        self.configuration.get('feature_engineering', {}).get('unbalance_treatment', None)

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
    print(base.method_feature_preprocessing)
    print(base.method_feature_preprocessing)
    