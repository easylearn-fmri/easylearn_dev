#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Base class for all modules
"""

import json
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectPercentile, SelectKBest, SelectFromModel, f_classif,f_regression, RFE,RFECV, VarianceThreshold, mutual_info_classif, SelectFromModel
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoCV, RidgeCV, RidgeClassifier, BayesianRidge, ElasticNetCV
from sklearn.gaussian_process import  GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.model_selection import KFold, StratifiedKFold,  ShuffleSplit


class BaseMachineLearning:

    def __init__(self):
        pass

    def get_configuration_(self, configuration_file):
        """Parse the configuration file
        """

        with open(configuration_file, 'r', encoding='utf-8') as config:
                    configuration = config.read()
        self.configuration = json.loads(configuration)

        return self

    def load_data(self):
        self.data_loading = self.configuration.get('data_loading', None)
        self.get_model_evaluation_parameters()
        
        # Get selected datasets
        for key in self.param_model_evaluation.items():
            if "selected_datasets" in key[0].split("__")[1]:
                selected_datasets_name = key[1]
        
        selected_datasets = []
        for g in selected_datasets_name:
            selected_datasets.append(self.data_loading.get(g.split(":")[0]).get(g.split(":")[1]))
        

    def get_preprocessing_parameters(self):
        self.method_feature_preprocessing = None
        self.param_feature_preprocessing = {}
                
        feature_preprocessing = self.configuration.get('feature_engineering', {}).get('feature_preprocessing', None)
        if feature_preprocessing and (list(feature_preprocessing.keys())[0] != 'None'):
            self.method_feature_preprocessing = [eval(list(feature_preprocessing.keys())[0] if list(feature_preprocessing.keys())[0] != 'None' else None)]
    
            for key in feature_preprocessing.keys():
                for key_ in feature_preprocessing.get(key).keys():
                    if key_ != []:
                        for key__ in feature_preprocessing.get(key).get(key_).keys():

                            param = feature_preprocessing.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            if self.criteria_of_eval_parameters(param):
                                param = eval(param)
                            self.param_feature_preprocessing.update({"feature_preprocessing__"+key_: [param]})

        self.param_feature_preprocessing = None if self.param_feature_preprocessing == {} else self.param_feature_preprocessing
             
        return self

    def get_dimension_reduction_parameters(self):
        self.method_dim_reduction = None
        self.param_dim_reduction = {}
                
        dimension_reduction = self.configuration.get('feature_engineering', {}).get('dimreduction', None)
        if dimension_reduction and (list(dimension_reduction.keys())[0] != 'None'):
            self.method_dim_reduction = [eval(list(dimension_reduction.keys())[0] if list(dimension_reduction.keys())[0] != 'None' else None)]
    
            for key in dimension_reduction.keys():
                for key_ in dimension_reduction.get(key).keys():
                    if key_ != []:
                        for key__ in dimension_reduction.get(key).get(key_).keys():

                            param = dimension_reduction.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            if self.criteria_of_eval_parameters(param):
                                param = eval(param)
                            if not (isinstance(param, list) or isinstance(param, tuple)):
                                param = [param]
                            self.param_dim_reduction.update({"dim_reduction__"+key_: param})
             
        self.param_dim_reduction = None if self.param_dim_reduction == {} else self.param_dim_reduction
        return self
        

    def get_feature_selection_parameters(self):
        self.method_feature_selection = None
        self.param_feature_selection = {}
        
        
        feature_selection = self.configuration.get('feature_engineering', {}).get('feature_selection', None)
        if feature_selection and (list(feature_selection.keys())[0] != 'None'):
            
            for key in feature_selection.keys():
                for key_ in feature_selection.get(key).keys():
                    if key_ != []:
                        for key__ in feature_selection.get(key).get(key_).keys():

                            param = feature_selection.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            if self.criteria_of_eval_parameters(param):
                                param = eval(param)
                            if not (isinstance(param, list) or isinstance(param, tuple)):
                                param = [param]
                            self.param_feature_selection.update({"feature_selection__"+key_:param})

            # Methods
            self.method_feature_selection = list(feature_selection.keys())[0] if list(feature_selection.keys())[0] != 'None' else None
            # Update point
            if self.method_feature_selection == 'RFECV()':
                self.method_feature_selection = 'RFECV(estimator=LinearSVC())' 
            
            if self.method_feature_selection == 'SelectFromModel(LassoCV())':
                self.method_feature_selection = f'SelectFromModel(LassoCV())'  
                self.param_feature_selection = None
            
            if self.method_feature_selection == 'SelectFromModel(ElasticNetCV())':
                self.method_feature_selection = 'SelectFromModel(ElasticNetCV('
                for keys in list(self.param_feature_selection.keys()):
                    param_ = keys.split('__')[1]
                    value_ = self.param_feature_selection[keys]
                    self.method_feature_selection = self.method_feature_selection+ f'{param_}={value_},'  
                self.method_feature_selection = self.method_feature_selection + '))'
                self.param_feature_selection = None
                
            self.method_feature_selection = [eval(self.method_feature_selection)]
        
        self.param_feature_selection = None if self.param_feature_selection == {} else self.param_feature_selection
        return self

    def get_unbalance_treatment_parameters(self):
        self.method_unbalance_treatment = None
        self.param_unbalance_treatment = {}
        
        
        unbalance_treatment = self.configuration.get('feature_engineering', {}).get('unbalance_treatment', None)
        if unbalance_treatment and (list(unbalance_treatment.keys())[0] != 'None'):
            self.method_unbalance_treatment = [(list(unbalance_treatment.keys())[0] if list(unbalance_treatment.keys())[0] != 'None' else None)]
    
            for key in unbalance_treatment.keys():
                for key_ in unbalance_treatment.get(key).keys():
                    if key_ != []:
                        for key__ in unbalance_treatment.get(key).get(key_).keys():

                            param = unbalance_treatment.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            if self.criteria_of_eval_parameters(param):
                                param = eval(param)
                            if not (isinstance(param, list) or isinstance(param, tuple)):
                                param = [param]
                            self.param_unbalance_treatment.update({"unbalance_treatment__"+key_:param})
             
        self.param_unbalance_treatment = None if self.param_unbalance_treatment == {} else self.param_unbalance_treatment
        return self

    def get_machine_learning_parameters(self):
        self.method_machine_learning = None
        self.param_machine_learning = {}
        
        machine_learning = self.configuration.get('machine_learning', None)
        keys = machine_learning.keys()
        if len(keys) == []:
            raise ValueError("There is no keys for machine_learning")
        elif len(keys) > 1:
            raise RuntimeError("Currently, easylearn only supports one type of machine learning")
            
        for keys in machine_learning:
            machine_learning = machine_learning.get(keys, None)

        if machine_learning and (list(machine_learning.keys())[0] != 'None'):
            # TODO: This place will update for supportting multiple estimators
            self.method_machine_learning = [eval(list(machine_learning.keys())[0] if list(machine_learning.keys())[0] != 'None' else None)]
    
            for key in machine_learning.keys():
                for key_ in machine_learning.get(key).keys():
                    if key_ != []:
                        for key__ in machine_learning.get(key).get(key_).keys():

                            param = machine_learning.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            # for example, DecisionTreeClassifier(max_depth=1) is a parameter of AdaBoostClassifier()
                            # Because a [sklearn] object has a
                            if self.criteria_of_eval_parameters(param):
                                param = eval(param)
                            if not (isinstance(param, list) or isinstance(param, tuple)):
                                param = [param]
                            
                            # TODO: Design a method to set params
                            self.param_machine_learning.update({"estimator__"+key_: param})
             
        self.param_machine_learning = None if self.param_machine_learning == {} else self.param_machine_learning
        return self

    def get_model_evaluation_parameters(self):
        self.method_model_evaluation = None
        self.param_model_evaluation = {}
        
        model_evaluation = self.configuration.get('model_evaluation', {})
        if model_evaluation and (list(model_evaluation.keys())[0] != 'None'):
            self.method_model_evaluation = eval(list(model_evaluation.keys())[0] if list(model_evaluation.keys())[0] != 'None' else None)
    
            for key in model_evaluation.keys():
                for key_ in model_evaluation.get(key).keys():
                    if key_ != []:
                        for key__ in model_evaluation.get(key).get(key_).keys():
                            
                            param = model_evaluation.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            # for example, DecisionTreeClassifier(max_depth=1) is a parameter of AdaBoostClassifier()
                            # Because a [sklearn] object has a
                            if type(param) is str:  # selected_dataset is list
                                if self.criteria_of_eval_parameters(param):
                                    param = eval(param)
                            if not (isinstance(param, list) or isinstance(param, tuple)):
                                param = [param]
                            self.param_model_evaluation.update({"model_evaluation__"+key_: param})
             
        self.param_model_evaluation = None if self.param_model_evaluation == {} else self.param_model_evaluation
        return self

    def get_statistical_analysis_parameters(self):
        self.configuration.get('statistical_analysis', None)

    def get_visualization_parameters(self):
        self.configuration.get('visualization', None)

    @staticmethod
    def criteria_of_eval_parameters(param):
        iseval = (
                    (
                        bool(re.search(r'\d', param)) or 
                        (param == 'None') or 
                        (bool(re.search(r'\(', param)) and bool(re.search(r'\)', param))) or
                        param == "None"
                    ) and
                    (
                        param != 'l1' and param != 'l2'
                    )
        )
        return iseval

if __name__ == '__main__':
    base = BaseMachineLearning()
    base.get_configuration_(configuration_file=r'F:\Python378\Lib\site-packages\eslearn\GUI\test\configuration_file.json')
    base.load_data()
    base.get_preprocessing_parameters()
    base.get_dimension_reduction_parameters()
    base.get_feature_selection_parameters()
    base.get_unbalance_treatment_parameters()
    base.get_machine_learning_parameters()
    base.get_model_evaluation_parameters()
    

    print(base.method_feature_preprocessing)
    print(base.param_feature_preprocessing)
    
    print(base.method_dim_reduction)
    print(base.param_dim_reduction)
    
    print(base.method_feature_selection)
    print(base.param_feature_selection)
    
    print(base.method_unbalance_treatment)
    print(base.param_unbalance_treatment)
    
    print(base.method_machine_learning)
    print(base.param_machine_learning)

    print(base.method_model_evaluation)
    print(base.param_model_evaluation)

    