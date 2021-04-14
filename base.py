#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Base class for all modules
"""

import time
import json
import re
import copy
import numpy as np
import pandas as pd
import os
import nibabel as nib
from scipy import io
from joblib import Memory
import abc
from abc import abstractmethod, ABCMeta
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SMOTENC)
from imblearn.under_sampling import (RandomUnderSampler,
                                    ClusterCentroids, 
                                    NearMiss,
                                    InstanceHardnessThreshold,
                                    CondensedNearestNeighbour,
                                    EditedNearestNeighbours,
                                    RepeatedEditedNearestNeighbours,
                                    AllKNN,
                                    NeighbourhoodCleaningRule,
                                    OneSidedSelection)

from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import make_scorer, accuracy_score, auc, f1_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectPercentile, SelectKBest, SelectFromModel, f_classif,f_regression, RFE,RFECV, VarianceThreshold, mutual_info_classif, SelectFromModel
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.linear_model import (LinearRegression, LogisticRegression, Lasso, 
                                  LassoCV, RidgeCV, Ridge,
                                  RidgeClassifier, BayesianRidge, ElasticNetCV
)

from sklearn.gaussian_process import  GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.model_selection import KFold, StratifiedKFold,  ShuffleSplit
from sklearn.pipeline import Pipeline


class BaseMachineLearning(object):
    """Base class for all machine learning

    Parameters:
    ----------
    configuration_file: file string
        configuration file containing all inputs

    Attributes:
    ----------
    method_feature_preprocessing_: list of sklearn object or None
    param_feature_preprocessing_: list of sklearn object or None
    
    method_dim_reduction_: list of sklearn object or None
    param_dim_reduction_: list of sklearn object or None
    
    method_feature_selection_: list of sklearn object or None
    param_feature_selection_: list of sklearn object or None
    
    method_unbalance_treatment_: list of sklearn object or None
    param_unbalance_treatment_: list of sklearn object or None
    
    machine_learning_type_: str
    method_machine_learning_: list of sklearn object or None
    param_machine_learning_: list of sklearn object or None

    method_model_evaluation_: list of sklearn object or None
    param_model_evaluation_: list of sklearn object or None

    model_: machine learning model, e.g., sklearn gridSearch model
    param_search_: parameter for search of machine learning pipeline
    """

    def __init__(self, configuration_file):
        self.configuration_file = configuration_file
        self._random_state = 0
        self._gridcv_k = 3
        self._search_strategy = "grid"
        self._n_jobs = 2

    def get_configuration_(self):
        """Get and parse the configuration file
        """

        with open(self.configuration_file, 'r', encoding='utf-8') as config:
                    configuration = config.read()
        self.configuration = json.loads(configuration)
        return self

    def get_preprocessing_parameters(self):
        self.method_feature_preprocessing_ = None
        self.param_feature_preprocessing_ = {}
                
        feature_preprocessing = self.configuration.get('feature_engineering', {}).get('feature_preprocessing', None)
        if feature_preprocessing and (list(feature_preprocessing.keys())[0] != 'None'):
            self.method_feature_preprocessing_ = [list(feature_preprocessing.keys())[0] if list(feature_preprocessing.keys())[0] != 'None' else None]
            self.method_feature_preprocessing_ = [self.security_eval(self.method_feature_preprocessing_[0])]
    
            for key in feature_preprocessing.keys():
                for key_ in feature_preprocessing.get(key).keys():
                    if key_ != []:
                        for key__ in feature_preprocessing.get(key).get(key_).keys():

                            param = feature_preprocessing.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            param = self.security_eval(param)
                            self.param_feature_preprocessing_.update({"feature_preprocessing__"+key_: [param]})

        # Fix the random_state for replication of results
        if self.method_feature_preprocessing_ and ("random_state" in self.method_feature_preprocessing_[0].get_params().keys()):
            self.param_feature_preprocessing_.update({"feature_preprocessing__"+'random_state': [self._random_state]})
        self.param_feature_preprocessing_ = None if self.param_feature_preprocessing_ == {} else self.param_feature_preprocessing_
             
        return self

    def get_dimension_reduction_parameters(self):
        self.method_dim_reduction_ = None
        self.param_dim_reduction_ = {}
                
        dimension_reduction = self.configuration.get('feature_engineering', {}).get('dimreduction', None)
        if dimension_reduction and (list(dimension_reduction.keys())[0] != 'None'):
            self.method_dim_reduction_ = [self.security_eval(list(dimension_reduction.keys())[0] if list(dimension_reduction.keys())[0] != 'None' else None)]
    
            for key in dimension_reduction.keys():
                for key_ in dimension_reduction.get(key).keys():
                    if key_ != []:
                        for key__ in dimension_reduction.get(key).get(key_).keys():
                            param = dimension_reduction.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            param = self.security_eval(param)
                            if not isinstance(param, (list, tuple)):
                                param = [param]
                            self.param_dim_reduction_.update({"dim_reduction__"+key_: param})
        
        # Fix the random_state for replication of results
        if self.method_dim_reduction_ and ("random_state" in self.method_dim_reduction_[0].get_params().keys()):
            self.param_dim_reduction_.update({"dim_reduction__"+'random_state': [self._random_state]})
        self.param_dim_reduction_ = None if self.param_dim_reduction_ == {} else self.param_dim_reduction_
        return self  

    def get_feature_selection_parameters(self):
        self.method_feature_selection_ = None
        self.param_feature_selection_ = {}
        
        feature_selection = self.configuration.get('feature_engineering', {}).get('feature_selection', None)
        if feature_selection and (list(feature_selection.keys())[0] != 'None'): 
            for key in feature_selection.keys():
                for key_ in feature_selection.get(key).keys():
                    if key_ != []:
                        for key__ in feature_selection.get(key).get(key_).keys():
                            param = feature_selection.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            param = self.security_eval(param)
                            if not isinstance(param, (list, tuple)):
                                param = [param]
                            self.param_feature_selection_.update({"feature_selection__"+key_:param})

            # Methods
            self.method_feature_selection_ = list(feature_selection.keys())[0] if list(feature_selection.keys())[0] != 'None' else None
            # Update point
            if "RFE" in self.method_feature_selection_:
                self.method_feature_selection_ = "RFE(estimator=SVC(kernel='linear'))"
            
            if self.method_feature_selection_ == 'SelectFromModel(LassoCV())':
                self.param_feature_selection_ = None
            
            if self.method_feature_selection_ == 'SelectFromModel(ElasticNetCV())':
                self.method_feature_selection_ = 'SelectFromModel(ElasticNetCV('
                for keys in list(self.param_feature_selection_.keys()):
                    param_ = keys.split('__')[1]
                    value_ = self.param_feature_selection_[keys]
                    self.method_feature_selection_ = self.method_feature_selection_+ f'{param_}={value_},'  
                self.method_feature_selection_ = self.method_feature_selection_ + '))'
                self.param_feature_selection_ = None
                
            self.method_feature_selection_ = [self.security_eval(self.method_feature_selection_)]
        
        # Fix the random_state for replication of results
        if self.method_feature_selection_ and ("random_state" in self.method_feature_selection_[0].get_params().keys()):
            self.param_feature_selection_.update({"feature_selection__"+'random_state': [self._random_state]})
        self.param_feature_selection_ = None if self.param_feature_selection_ == {} else self.param_feature_selection_
        return self

    def get_unbalance_treatment_parameters(self):
        self.method_unbalance_treatment_ = None
        self.param_unbalance_treatment_ = {}

        unbalance_treatment = self.configuration.get('feature_engineering', {}).get('unbalance_treatment', None)
        if unbalance_treatment and (list(unbalance_treatment.keys())[0] != 'None'):
            self.method_unbalance_treatment_ = (self.security_eval(list(unbalance_treatment.keys())[0]) if list(unbalance_treatment.keys())[0] != 'None' else None)
    
            for key in unbalance_treatment.keys():
                for key_ in unbalance_treatment.get(key).keys():
                    if key_ != []:
                        for key__ in unbalance_treatment.get(key).get(key_).keys():

                            param = unbalance_treatment.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            param = self.security_eval(param)
                            if not isinstance(param, (list, tuple)):
                                param = [param]
                            self.param_unbalance_treatment_.update({"unbalance_treatment__"+key_:param})
                     
        # Fix the random_state for replication of results
        if self.method_unbalance_treatment_ and "random_state" in self.method_unbalance_treatment_.get_params().keys():
            self.method_unbalance_treatment_.set_params(**{"random_state": self._random_state})
            self.param_unbalance_treatment_.update({"unbalance_treatment__"+'random_state': [self._random_state]})
        self.param_unbalance_treatment_ = None if self.param_unbalance_treatment_ == {} else self.param_unbalance_treatment_
        
        return self

    def get_machine_learning_parameters(self):
        self.method_machine_learning_ = None
        self.param_machine_learning_ = {}
        
        machine_learning = self.configuration.get('machine_learning', None)
        self.machine_learning_type_ = list(machine_learning.keys()) if machine_learning else None
        if self.machine_learning_type_ is None:
            raise ValueError("There is no keys for machine_learning")
        elif len(self.machine_learning_type_) > 1:
            raise RuntimeError("Currently, easylearn only supports one type of machine learning")
            
        for keys in machine_learning:
            machine_learning = machine_learning.get(keys, None)

        if machine_learning and (list(machine_learning.keys())[0] != 'None'):
            # TODO: This place will update for supporting multiple estimators
            self.method_machine_learning_ = [self.security_eval(list(machine_learning.keys())[0] if list(machine_learning.keys())[0] != 'None' else None)]
    
            for key in machine_learning.keys():
                for key_ in machine_learning.get(key).keys():
                    if key_ != []:
                        for key__ in machine_learning.get(key).get(key_).keys():

                            param = machine_learning.get(key).get(key_).get(key__)
                            param = 'None' if param == '' else param
                            # Parse parameters: if param is digits str or containing "(" and ")", we will eval the param
                            # for example, DecisionTreeClassifier(max_depth=1) is a parameter of AdaBoostClassifier()
                            # Because a [sklearn] object has a
                            param = self.security_eval(param)
                            if not isinstance(param, (list, tuple)):
                                param = [param]
                            # TODO: Design a method to set params
                            self.param_machine_learning_.update({"estimator__"+key_: param})
         
        # Fix the random_state for replication of results
        if self.method_machine_learning_ and "random_state" in self.method_machine_learning_[0].get_params().keys():
            self.param_machine_learning_.update({"estimator__"+'random_state': [self._random_state]})
        self.param_machine_learning_ = None if self.param_machine_learning_ == {} else self.param_machine_learning_
        return self

    def get_model_evaluation_parameters(self):
        self.method_model_evaluation_ = None
        self.param_model_evaluation_ = {}
        self.statistical_analysis = self.configuration.get('model_evaluation', {}).get("Statistical_analysis", None)
        if self.statistical_analysis:
            self.configuration.get('model_evaluation', {}).pop("Statistical_analysis")
        model_evaluation = self.configuration.get('model_evaluation', None)
        
        if model_evaluation and (list(model_evaluation.keys())[0] != 'None'):
            self.method_model_evaluation_ = list(model_evaluation.keys())[0] if list(model_evaluation.keys())[0] != 'None' else None
            
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
                                    param = self.security_eval(param)
                            self.param_model_evaluation_.update({key_: param})
             
            # ------Give parameter to method------
            pme = ""
            ik_end = len(self.param_model_evaluation_)  - 1
            for ik, key_pme in enumerate(self.param_model_evaluation_):
                if ik != ik_end:
                    pme = pme + f"{key_pme}={self.param_model_evaluation_[key_pme]}" + ", "
                else:
                    pme = pme + f"{key_pme}={self.param_model_evaluation_[key_pme]}"
            
            self.method_model_evaluation_ = self.method_model_evaluation_.split("(")[0] + "(" + pme + self.method_model_evaluation_.split("(")[1]
            self.method_model_evaluation_ = self.security_eval(self.method_model_evaluation_)
            
        return self

    def get_statistical_analysis_parameters(self):
        self.method_statistical_analysis_ = list(self.statistical_analysis.keys())[0]
        # parameters
        # FIX the logical and extending to...
        self.param_statistical_analysis_ = None
        for key1 in self.statistical_analysis.keys():
            for key2 in self.statistical_analysis.get(key1, {}).keys():
                for key3 in self.statistical_analysis[key1].get(key2, {}).keys():
                    self.param_statistical_analysis_ = self.statistical_analysis.get(key1, {}).get(key2, {}).get(key3, {})
                    self.param_statistical_analysis_ = self.security_eval(self.param_statistical_analysis_)
                
        return self

    def get_visualization_parameters(self):
        self.configuration.get('visualization', None)

    def get_all_inputs(self):
        self.get_configuration_()
        self.get_preprocessing_parameters()
        self.get_dimension_reduction_parameters()
        self.get_feature_selection_parameters()
        self.get_unbalance_treatment_parameters()
        self.get_machine_learning_parameters()
        self.get_model_evaluation_parameters()
        self.get_statistical_analysis_parameters()
        self.make_sklearn_search_model_()
        return self


    @staticmethod
    def security_eval(expression):
        """Security evaluation of python expression
        
        FIX: 'eval' had security hole
        """
        
        iseval = (
                    (
                        bool(re.search(r'\d', expression)) or 
                        (expression == 'None') or 
                        (bool(re.search(r'\(', expression)) and bool(re.search(r'\)', expression))) 
                    ) 
                    and
                    (
                        expression != 'l1' and
                        expression != 'l2'
                        # not bool(re.search('del',  expression)) and
                        # not bool(re.search('open',  expression)) and
                        # not bool(re.search('move',  expression)) and
                        # not bool(re.search('copy',  expression))     
                    )
        )

        if iseval:
            evaluated_expression = eval(expression)
        else:
            evaluated_expression = expression

        return evaluated_expression

    def make_sklearn_search_model_(self, metric=accuracy_score):
        
        """Construct pipeline_

        Currently, the pipeline_ only supports one specific method for corresponding method, 
        e.g., only supports one dimension reduction method for dimension reduction.
        In the next version, the pipeline_ will support multiple methods for each corresponding method.
        
        Parameters:
        ----------
        metric: sklearn metric object, such as accuracy_score, auc, f1_score. Default is accuracy_score
            Metric is used evaluate model using cross validation in search strategy, such as GridSearchCV.

        Returns:
        -------
        model_
        """
        
        self.memory = Memory(location=os.path.dirname(self.configuration_file), verbose=False)

        # Construct sklearn pipeline
        self.pipeline_ = Pipeline(steps=[
            ('feature_preprocessing','passthrough'),
            ('dim_reduction', 'passthrough'),
            ('feature_selection', 'passthrough'),
            ('estimator', 'passthrough'),
            ], 
            memory=self.memory
        )

        # Set parameters of search CV
        self.param_search_ = {}
        if self.method_feature_preprocessing_:
            self.param_search_.update({'feature_preprocessing':self.method_feature_preprocessing_})
        if self.param_feature_preprocessing_:   
            self.param_search_.update(self.param_feature_preprocessing_)
            
        if self.method_dim_reduction_:
            self.param_search_.update({'dim_reduction':self.method_dim_reduction_})
        if self.param_dim_reduction_:
            self.param_search_.update(self.param_dim_reduction_)
                
        if self.method_feature_selection_:
            self.param_search_.update({'feature_selection': self.method_feature_selection_})
        if self.param_feature_selection_:
            self.param_search_.update(self.param_feature_selection_)
            
        if self.method_machine_learning_:
            self.param_search_.update({'estimator': self.method_machine_learning_})
        if self.param_machine_learning_:
            self.param_search_.update(self.param_machine_learning_)
        
        # If no parameters' length greater than 1, using sklearn pipeline for speed up, instead of GridSearchCV or RandomizedSearchCV.
        self.is_search = self.get_is_search(self.param_search_)
        if not self.is_search:
            if self.method_feature_preprocessing_:
                self.pipeline_.set_params(**{'feature_preprocessing':self.method_feature_preprocessing_[0]})
            if self.param_feature_preprocessing_:   
                mapping = self.parse_search_params(self.param_feature_preprocessing_)
                self.pipeline_['feature_preprocessing'].set_params(**mapping)
                
            if self.method_dim_reduction_:
                self.pipeline_.set_params(**{'dim_reduction':self.method_dim_reduction_[0]})
            if self.param_dim_reduction_:
                mapping = self.parse_search_params(self.param_dim_reduction_)
                self.pipeline_['dim_reduction'].set_params(**mapping)   
                
            if self.method_feature_selection_:
                self.pipeline_.set_params(**{'feature_selection': self.method_feature_selection_[0]})
            if self.param_feature_selection_:
                mapping = self.parse_search_params(self.param_feature_selection_)
                self.pipeline_['feature_selection'].set_params(**mapping)

            if self.method_machine_learning_:
                self.pipeline_.set_params(**{'estimator': self.method_machine_learning_[0]})
            if self.param_machine_learning_:
                mapping = self.parse_search_params(self.param_machine_learning_)
                self.pipeline_['estimator'].set_params(**mapping)

        # Building model
        if "Classification" in self.machine_learning_type_:
            cv = StratifiedKFold(n_splits=self._gridcv_k, random_state=self._random_state, shuffle=True)  # Default is StratifiedKFold
        else:
            cv = KFold(n_splits=self._gridcv_k, random_state=self._random_state, shuffle=True)
        
        if self.is_search:
            if self._search_strategy == 'grid':
                self.model_ = GridSearchCV(
                    self.pipeline_, n_jobs=self._n_jobs, param_grid=self.param_search_, cv=cv, 
                    scoring = make_scorer(metric), refit=True
                )
            elif self._search_strategy == 'random':
                self.model_ = RandomizedSearchCV(
                    self.pipeline_, n_jobs=self._n_jobs, param_distributions=self.param_search_, cv=cv, 
                    scoring = make_scorer(metric), refit=True, n_iter=self.n_iter_of_randomedsearch,
                )
            else:
                print("Please specify which search strategy!\n")
                return
        else:
            self.model_ = self.pipeline_

        return self
    
    @staticmethod
    def get_is_search(dictionary):
        """ Identify whether search params (grid search or random search) or just using pipeline
        """
        
        is_search = False
        for key in dictionary:
            if dictionary[key] and len(dictionary[key]) > 1:
                is_search = True
                break
        return is_search
    
    @staticmethod
    def parse_search_params(dictionary):
        """ When just using pipeline and not search parameters
        I use 'set_params' to set parameters for pipeline to save running time.       
        """

        mapping = {}
        for key in dictionary:
            mapping.update({key.split("__")[1]:dictionary[key][0]})
        return mapping
            
    def save_weight(self, weights=None, out_dir=None):
        """Save contribution weight of features for each modality
        
        Parameters:
        ----------
        
        weights: list of numpy.ndarray
            Contribution weights of each fold of each modality (e.g. 5-fold cross validation)

        out_dir: str
            Output directory

        Returns:
        -------
        None
        """

        mean_wei = np.reshape(np.mean(weights, axis=0), [-1,])
        
        for group in self.mask_:
            loc_start = 0
            for im, modality in enumerate(self.mask_[group]):
                mask = self.mask_[group][modality]
                mean_weight = np.zeros(mask.shape)

                # Updating mask location index
                n_features = mask.sum()
                loc_end = loc_start + n_features

                mean_weight[mask] = mean_wei[loc_start:loc_end]

                # Save
                if self.data_format_[group][modality] in ["nii","gz"]:
                    out_name_wei = os.path.join(out_dir, f"weight_{modality}.nii.gz")
                    if os.path.exists(out_name_wei):
                        time_ = time.strftime('%Y%m%d%H%M%S')
                        out_name_wei = os.path.join(out_dir, f"weight_{modality}_{time_}.nii.gz")
                    mean_weight2nii = nib.Nifti1Image(mean_weight, self.affine_[group][modality])
                    mean_weight2nii.to_filename(out_name_wei)
                else:
                    out_name_wei = os.path.join(out_dir, f"weight_{modality}.csv")
                    if os.path.exists(out_name_wei):
                        time_ = time.strftime('%Y%m%d%H%M%S')
                        out_name_wei = os.path.join(out_dir, f"weight_{modality}_{time_}.csv")
                        
                    if len(np.shape(mean_weight)) > 1:
                        np.savetxt(out_name_wei, mean_weight, delimiter=',')  
                    else:
                        pd.Series(mean_weight).to_csv(out_name_wei, header=False)
                
                # Updating mask location index
                loc_start += n_features  # n_features in this point is the value in previous iteration
                
            break  # Assuming the size of the same modality in different group are matching for each other


# TODO
class MakeModel():
    """Make a machine learning model

    make_sklearn_search_model_ should be integrated into the this class

    Parameters:
    ----------
    pass

    Attributes:
    ----------
    fit:
    predict:
    
    returns:
    -------
    pass
    """

    def __init__(self):
        baseml = BaseMachineLearning()

    def make_sklearn_search_model_(self, metric=accuracy_score):
        
        """Construct pipeline_

        Currently, the pipeline_ only supports one specific method for corresponding method, 
        e.g., only supports one dimension reduction method for dimension reduction.
        In the next version, the pipeline_ will support multiple methods for each corresponding method.
        
        Parameters:
        ----------
        metric: sklearn metric object, such as accuracy_score, auc, f1_score. Default is accuracy_score
            Metric is used evaluate model using cross validation in search strategy, such as GridSearchCV.

        Returns:
        -------
        model_
        """
        
        self.memory = Memory(location=os.path.dirname(self.configuration_file), verbose=False)

        # Construct sklearn pipeline
        self.pipeline_ = Pipeline(steps=[
            ('feature_preprocessing','passthrough'),
            ('dim_reduction', 'passthrough'),
            ('feature_selection', 'passthrough'),
            ('estimator', 'passthrough'),
            ], 
            memory=self.memory
        )

        # Set parameters of search CV
        self.param_search_ = {}
        if baseml.method_feature_preprocessing_:
            self.param_search_.update({'feature_preprocessing':self.method_feature_preprocessing_})
        if baseml.param_feature_preprocessing_:   
            self.param_search_.update(self.param_feature_preprocessing_)
            
        if baseml.method_dim_reduction_:
            self.param_search_.update({'dim_reduction':self.method_dim_reduction_})
        if baseml.param_dim_reduction_:
            self.param_search_.update(self.param_dim_reduction_)
                
        if baseml.method_feature_selection_:
            self.param_search_.update({'feature_selection': self.method_feature_selection_})
        if baseml.param_feature_selection_:
            self.param_search_.update(self.param_feature_selection_)
            
        if baseml.method_machine_learning_:
            self.param_search_.update({'estimator': self.method_machine_learning_})
        if baseml.param_machine_learning_:
            self.param_search_.update(self.param_machine_learning_)
        
        # If no parameters' length greater than 1, using sklearn pipeline for speed up, instead of GridSearchCV or RandomizedSearchCV.
        self.is_search = self.get_is_search(self.param_search_)
        if not self.is_search:
            if self.method_feature_preprocessing_:
                self.pipeline_.set_params(**{'feature_preprocessing':self.method_feature_preprocessing_[0]})
            if self.param_feature_preprocessing_:   
                mapping = self.parse_search_params(self.param_feature_preprocessing_)
                self.pipeline_['feature_preprocessing'].set_params(**mapping)
                
            if self.method_dim_reduction_:
                self.pipeline_.set_params(**{'dim_reduction':self.method_dim_reduction_[0]})
            if self.param_dim_reduction_:
                mapping = self.parse_search_params(self.param_dim_reduction_)
                self.pipeline_['dim_reduction'].set_params(**mapping)   
                
            if self.method_feature_selection_:
                self.pipeline_.set_params(**{'feature_selection': self.method_feature_selection_[0]})
            if self.param_feature_selection_:
                mapping = self.parse_search_params(self.param_feature_selection_)
                self.pipeline_['feature_selection'].set_params(**mapping)

            if self.method_machine_learning_:
                self.pipeline_.set_params(**{'estimator': self.method_machine_learning_[0]})
            if self.param_machine_learning_:
                mapping = self.parse_search_params(self.param_machine_learning_)
                self.pipeline_['estimator'].set_params(**mapping)

        # Building model
        if "Classification" in list(self.configuration.get("machine_learning").keys()):
            cv = StratifiedKFold(n_splits=self._gridcv_k, random_state=self._random_state, shuffle=True)  # Default is StratifiedKFold
        else:
            cv = KFold(n_splits=self._gridcv_k, random_state=self._random_state, shuffle=True)  # Default is StratifiedKFold
        
        if self.is_search:
            if self._search_strategy == 'grid':
                self.model_ = GridSearchCV(
                    self.pipeline_, n_jobs=self._n_jobs, param_grid=self.param_search_, cv=cv, 
                    scoring = make_scorer(metric), refit=True
                )
            elif self._search_strategy == 'random':
                self.model_ = RandomizedSearchCV(
                    self.pipeline_, n_jobs=self._n_jobs, param_distributions=self.param_search_, cv=cv, 
                    scoring = make_scorer(metric), refit=True, n_iter=self.n_iter_of_randomedsearch,
                )
            else:
                print("Please specify which search strategy!\n")
                return
        else:
            self.model_ = self.pipeline_

        return self


#%% ==========================================================================
class DataLoader():
    """Load datasets according to different data types and handle extreme values

    Parameters:
    ----------
    configuration_file: file string
        configuration file containing all inputs

    Attributes:
    ----------
    targets_: ndarray of shape (n_samples, )
    
    features_: ndarray of shape (n_samples, n_features) 

    mask_: dictionary, each element contains a mask of a modality of a group
    
    data_format_: str, data format such as 'nii', 'mat'
    
    affine_: 4 by 4 matrix, image affine

    id_: subject id

    Notes:
    -----
    1. Easylearn allows users to input multiple modalities for one group. 
       Then, easylearn will feed features combined multiple modalities into machine learning model.
    
    2. If there is only one input file for one modality, 
       then data in this file should have a column of "__ID__" (unique idenfity), 
       otherwise easylearn will take the first column as "__ID__".
       So that easylearn can match cases between modalities and match modalities with targets and covariates.
       If this file
    
    3. If ther are multiple input files for one modality, then the files name must contain r'.*(sub.?[0-9].*).*' for 
       extracting unique idenfity information to match like above. 
       For example one file name contains strings of "sub-008.nii".
    
    4. Easylearn only allows users to input targets as one integer by type in the GUI 
       (only for classification) or a file path for one group. 
       If the input targets is a file, then data in the file should have a column of "__ID__" and a column of "__Targets__", 
       otherwise easylearn will take the first column as "__ID__", and the second as "__Targets__".
       
       If the input targets is a integer(only for classification) , 
       then easylearn will assign the integer as target for all case in the group.

    5. Easylearn allows users to input covariates, such as age, gender. Now, only file can be input as the covariates.
       If user given easylearn a covariates file, then data in the file should have a column of "__ID__", 
       otherwise easylearn will take the first column as "__ID__".

    """
    
    def __init__(self, configuration_file):
        self.configuration_file = configuration_file
        
        # Generate type2fun dictionary
        # TODO: Extended to handle other formats
        self.type2fun = {
                    ".nii": self.read_nii, 
                    ".img": self.read_nii,
                    ".mat": self.read_mat, 
                    ".txt": self.read_csv,
                    ".csv": self.read_csv,
                    ".xlsx": self.read_excel,
                    ".xls": self.read_excel,
                    ".npy": self.read_ndarray,
        }
    
    def get_configuration_(self):
        """Get and parse the configuration file
        """

        with open(self.configuration_file, 'r', encoding='utf-8') as config:
                    configuration = config.read()
        self.configuration = json.loads(configuration)
        return self

    def load_data(self):
        self.get_configuration_()
        load_data = self.configuration.get('data_loading', None)
        
        #%% ==========================================Check datasets=================================
        # NOTE.: That check whether the feature dimensions of the same modalities in different groups are equal
        # is placed in the next section.
        targets = {}
        self.covariates_ = {}
        for i, gk in enumerate(load_data.keys()):
            
            # Check the number of modality across all group is equal
            if i == 0:
                n_mod = len(load_data.get(gk).get("modalities").keys())
            else:
                if n_mod != len(load_data.get(gk).get("modalities").keys()):
                    raise ValueError("The number of modalities in each group is not equal, check your inputs")
                    return
                n_mod = len(load_data.get(gk).get("modalities").keys())
                
            # Get targets
            targets_input = load_data.get(gk).get("targets")
            targets[gk] = self.read_targets(targets_input)  

    
            # Get covariates
            covariates_input = load_data.get(gk).get("covariates")
            if (isinstance(covariates_input, str) and
                covariates_input.strip() != "" and
                (not os.path.isfile(covariates_input))):  # Easylearn only supports file input for covariates
                raise ValueError("Easylearn only supports file input for covariates, check your covariates for {gk}")
            self.covariates_[gk] = self.base_read(covariates_input)
            
            # Check the number of files in each modalities in the same group is equal
            for j, mk in enumerate(load_data.get(gk).get("modalities").keys()):
                modality = load_data.get(gk).get("modalities").get(mk)
                
                # Filses
                input_files = modality.get("file")
                if j == 0:
                    n_file = self.get_file_len(input_files)  # Initialize n_file
                else:
                    if n_file != self.get_file_len(input_files):  # Left is previous, right is current loop
                        raise ValueError(f"The number of files in each modalities in {gk} is not equal, check your inputs")
                        return
                n_file = self.get_file_len(input_files)  # Update n_file

                # Check the number of targets in each modalities is equal to the number of files          
                # If the type of targets is list, and number of files are not equal to targets, then raise error
                if (isinstance(targets[gk],list)) and (n_file != len(targets[gk])):
                    raise ValueError(f"The number of files in {mk} of {gk} is not equal to the number of targets, check your inputs")
                    return
        
                # Check the number of lines of covariates in each modalities is equal to the number of files
                # If covariates is not int (0), and number of files are not equal to covariates, then raise error
                if (not isinstance(self.covariates_[gk],int)) and (n_file != len(self.covariates_[gk])):
                    raise ValueError(f"The number of files in {mk} of {gk} is not equal to its' number of covariates, check your inputs")
                    return
                
        #%% ==========================================Get selected datasets =================================
        shape_of_data = {}
        feature_applied_mask_all = {}
        feature_applied_mask_and_add_otherinfo = {}
        col_drop = {}
        self.mask_ = {}
        self.data_format_ = {}
        self.affine_ =  {}

        for gi, gk in enumerate(load_data.keys()): 
            col_drop[gk] = ["__Targets__"]
            shape_of_data[gk] = {}
            feature_applied_mask_and_add_otherinfo[gk] = {}
            feature_applied_mask_all[gk] = {}
            self.mask_[gk] = {}
            self.data_format_[gk] = {}
            self.affine_[gk] = {}
            
            for jm, mk in enumerate(load_data.get(gk).get("modalities").keys()):
                modality = load_data.get(gk).get("modalities").get(mk)
               
                # Get files
                # If only input one file for one modality, 
                # then I think the file contained multiple cases' data
                input_files = modality.get("file")
                n_file = self.get_file_len(input_files)
                if len(input_files) == 1:
                    one_file_per_modality = True
                else:
                    one_file_per_modality = False
                
                # Get features' format and affine for each modality
                # I think all files in on modality are in the same format
                # So I take the first file in corresponding modality as example file
                # TODO: other situations
                self.data_format_[gk][mk], self.affine_[gk][mk] = self.get_data_format(input_files[0])

                # Get Features
                all_features = self.read_file(input_files, False)
                if one_file_per_modality:
                    all_features_ = list(all_features)[0]
                else:
                    all_features_ = False

                # Get cases' name (unique ID) in this modality
                # If one_file_per_modality = False, then each file name must contain r'.*(sub.?[0-9].*).*'
                # If one_file_per_modality = True and all_features_ is DataFrame, 
                # then the DataFrame must have header of "__ID__" which contain the unique_identifier,
                # otherwise easylearn will take the first column as "__ID__".
                if isinstance(all_features_, pd.core.frame.DataFrame) and ("__ID__" not in all_features_.columns):
                    # raise ValueError(f"The dataset of {input_files} did not have '__ID__' column, check your dataset")
                    unique_identifier = all_features_.iloc[:,0]  # Take the first column as __ID__
                    print(f"The dataset of {input_files} did not have '__ID__' column, easylearn take the first column as ID\n")
                elif isinstance(all_features_, pd.core.frame.DataFrame) and ("__ID__" in all_features_.columns):
                    unique_identifier = pd.DataFrame(all_features_["__ID__"])
                    all_features_.drop("__ID__", axis=1, inplace=True)
                    all_features = [all_features_]
                elif isinstance(all_features_, np.ndarray):
                    all_features_ = pd.DataFrame(all_features_)
                    all_features = [all_features_]
                    unique_identifier = pd.DataFrame(all_features_.iloc[:,0], dtype=np.str) # Take the first column as __ID__
                    unique_identifier.columns = ["__ID__"]
                    all_features = [all_features_.iloc[:,1:]]
                else:
                    unique_identifier = self.extract_id(input_files)  # Multiple files
                        
                # Apply mask to feature
                mask_input = modality.get("mask")
                # Do not extract triangule matrix when read mask file
                self.mask_[gk][mk] = self.base_read(mask_input)
                if not isinstance(self.mask_[gk][mk], int):  # If have mask
                   # TODO:  Allow uses to set threshold for mask, now code only set zero threshold.
                   self.mask_[gk][mk] = self.mask_[gk][mk] != 0 
                   # Apply mask
                   feature_applied_mask = [fa[self.mask_[gk][mk]] for fa in all_features]
                   feature_applied_mask = np.array(feature_applied_mask)
                else:
                   feature_applied_mask = [fa for fa in all_features]
                   feature_applied_mask, self.mask_[gk][mk] = self.get_upper_tri_mat(feature_applied_mask, one_file_per_modality)
                   feature_applied_mask = np.array(feature_applied_mask)
                   feature_applied_mask = feature_applied_mask.reshape(n_file,-1)

                # Concat feature across different modalities and groups 
                # In addition, the second and later modality are sorted according with the first one using pd.merge method
                feature_applied_mask = pd.concat([unique_identifier, pd.DataFrame(feature_applied_mask)], axis=1) 
                if jm == 0:
                    feature_applied_mask_all[gk] = feature_applied_mask
                else:
                    feature_applied_mask_all[gk] = pd.merge(feature_applied_mask_all[gk], feature_applied_mask, left_on="__ID__", right_on="__ID__")  

            # Dropout __ID__ from feature_applied_mask_all
            unique_identifier_ = feature_applied_mask_all[gk]["__ID__"]  # Get the final unique_identifier
            feature_applied_mask_all[gk].drop("__ID__", axis=1, inplace=True)
            
            #%% =====================Match targets and covariates with unique_identifier_==============================
            # NOTE. subj-name is come from the first modality due to the second and later modality are sorted 
            # according with the first one using pd.merge method 

            # Sort targets and check
            if (isinstance(targets[gk],int)):
                targets[gk] = [targets[gk] for ifile in range(n_file)]
                targets[gk] = pd.DataFrame(targets[gk])
                targets[gk]["__ID__"] = unique_identifier_
                targets[gk].rename(columns={0: "__Targets__"}, inplace=True)
            elif isinstance(targets[gk], pd.core.frame.DataFrame) and ("__ID__" not in targets[gk].columns):
                # raise ValueError(f"The targets of {gk} did not have '__ID__' column, check your targets") 
                print(f"The targets of {gk} did not have '__ID__' column, easylearn take the first column as ID\n") 
                # Take the first column as __ID__, and the second column as __Targets__
                targets[gk].columns = ["__ID__", "__Targets__"] 
            elif isinstance(targets[gk], np.ndarray):
                targets[gk] = pd.DataFrame(targets[gk])
                # Take the first column as __ID__, and the second as __Targets__
                targets[gk].rename(columns={0:"__ID__", 1:"__Targets__"}, inplace=True)
            
            targets[gk] = pd.merge(unique_identifier_, targets[gk], left_on="__ID__", right_on="__ID__", how='inner')
            if targets[gk].shape[0] != n_file:
                    raise ValueError(f"The subjects' ID in targets is not totally matched with its' data file name in {mk} of {gk} , check your ID in targets or check your data file name")

            # Sort covariates and check                
            if (not isinstance(self.covariates_[gk],int)):  # User have given covariates
                if isinstance(self.covariates_[gk], pd.core.frame.DataFrame) and ("__ID__" not in self.covariates_[gk].columns):
                    # raise ValueError(f"The covariates of {gk} did not have 'ID' column, check your covariates")
                    print(f"The covariates of {gk} did not have 'ID' column, easylearn take the first column as ID\n")
                    colname = list(self.covariates_[gk].columns)
                    colname[0] = "__ID__"
                    self.covariates_[gk].columns = colname
                elif isinstance(self.covariates_[gk], np.ndarray): 
                    self.covariates_[gk] = pd.DataFrame(self.covariates_[gk])
                    self.covariates_[gk].rename(columns={0:"__ID__"}, inplace=True) # Take the first column as __ID__
                
                self.covariates_[gk] = pd.merge(unique_identifier_, self.covariates_[gk], left_on="__ID__", right_on="__ID__") 
                if self.covariates_[gk].shape[0] != n_file:
                    raise ValueError(f"The subjects' ID in covariates is not totally matched with its' data file name in {mk} of {gk} , check your ID in covariates or check your data file name")
                        
            # Check whether the feature dimensions of the same modalities in different groups are equal
            shape_of_data[gk][mk] = feature_applied_mask_all[gk].shape
            if gi == 0:
                gk_pre = gk
            else:
                if shape_of_data[gk_pre][mk][-1] != shape_of_data[gk][mk][-1]:
                    raise ValueError(f"Feature dimension of {mk} in {gk_pre} is {shape_of_data[gk_pre][mk][-1]} which is not equal to {mk} in {gk}: {shape_of_data[gk][mk][-1]}, check your inputs")
             

            # Concat datasets across different group
            unique_identifier_ = pd.DataFrame([f"{gk}_{ui}" for ui in unique_identifier_])
            if gi == 0:
                self.id_ = unique_identifier_
                self.targets_ = targets[gk]["__Targets__"]
                self.features_ = feature_applied_mask_all[gk]
            else:
                self.id_ = pd.concat([self.id_, unique_identifier_])
                self.targets_ = pd.concat([self.targets_,  targets[gk]["__Targets__"]])
                self.features_ = pd.concat([self.features_, feature_applied_mask_all[gk]], axis=0)

        self.id_ = self.id_.values
        self.targets_ = np.float64(self.targets_.values)
        self.features_ = np.float64(self.features_.values)
        return self
       
    #%% ========================utilt functions========================
    def get_file_len(self, files):
        """If the files lenght is 1, then the length is the length of content of the files
        """
        
        file_len = len(files)
        if file_len == 1:
            all_features = self.read_file(files, False)
            all_features = [fe for fe in all_features][0]
            file_len = len(all_features)
        return file_len
        
    def del_id(self, all_features, input_files):
        """Delete "__ID__" in each DataFrame in all_features

        At last, the all_features contains only the feature

        Parameters:
        ----------
        all_features: list of DataFrames or ndarray
            All features

        input_files: list of
        """
        
        all_features_ = list()
        for df, file in zip(all_features, input_files):
            if isinstance(df, pd.core.frame.DataFrame) and ("__ID__" not in df.columns):
                raise ValueError(f"The dataset of {file} did not have '__ID__' column, check your dataset")
            elif isinstance(df, pd.core.frame.DataFrame) and ("__ID__" in df.columns):
                df.drop("__ID__", axis=1, inplace=True)
                all_features_.append(df)
            else:
                all_features_.append(df[:,1:])  # Take the first column as __ID__, and drop the first column.
                
        return all_features_

    def read_targets(self, targets_input):
        """Read targets from GUI

        Easylearn only supports one integer input or a file input for targets
        """

        if (targets_input == []) or (targets_input == ''):
            targets = None  # Clustering algorithm does not need targets
        
        if os.path.isfile(targets_input):
            targets = self.base_read(targets_input) 
        elif len(re.findall(r'[A-Za-z]', targets_input)):  # Contain alphabet
            raise ValueError(f"The '{targets_input}' is not exist OR The targets(labels) must be an Arabic numbers or file, but it contain alphabet, check your targets: '{targets_input}'")
            return
        elif ' ' in targets_input:
            targets = targets_input.split(' ')
            targets = [int(targets_) for targets_ in targets]
            if len(targets) > 1:
              raise ValueError(f"The targets(labels) must be an integer or a file, check your targets: {targets_input}\n")
        elif ',' in targets_input:
            targets = targets_input.split(',')
            targets = [int(targets_) for targets_ in targets]
            if len(targets) > 1:
              raise ValueError(f"The targets(labels) must be an integer or a file, check your targets: {targets_input}\n")
        else:
            targets = eval(targets_input)  
            if not isinstance(targets, int):
              raise ValueError(f"The targets(labels) must be an integer or a file for classification, check your targets: {targets_input}\n")      

        return targets

    def read_file(self, input_files, to1d=False):  
        """Read all input files
        """

        data = (self.base_read(file, to1d) for file in input_files)
        return data


    def base_read(self, file, to1d=False):
        """Read one file
        
        Parameters:
        ----------
        file: Path str
            input file of one case
        
        to1d: Bool
            whether transform data to 1 dimension
        
        Return:
        ------
        data: ndarray
        """
        
        if (file == []) or (file == ''):
            return 0
        elif not os.path.isfile(file):
            raise ValueError(f" Cannot find the file:'{file}'")
            return
        else:
            # Identify file type
            [path, filename] = os.path.split(file)
            suffix = os.path.splitext(filename)[-1]
            # Read
            data = self.type2fun[suffix](file)

            if to1d:
                data = np.reshape(data, [-1,])
                
        return data
    
    @staticmethod
    def get_data_format(example_file):
        """Get  features' data format and affine
        """

        # TODO: considering such as nii.gz in the future
        affine = None
        data_format = example_file.split(".")[-1]
        if data_format in ["nii","gz"]:
            obj = nib.load(example_file)
            affine = obj.affine
        return data_format, affine

    @ staticmethod
    def read_nii(file):      
        obj = nib.load(file)
        data = obj.get_fdata()
        return data

    @ staticmethod
    def read_mat(file):
        dataset_struct = io.loadmat(file)
        return dataset_struct[list(dataset_struct.keys())[3]]

    @ staticmethod
    def read_csv(file):
        data = pd.read_csv(file)                    
        return data

    @ staticmethod
    def read_excel(file):
        """
        Not consider symmetric matrix
        """
        
        data = pd.read_excel(file)
        return data
    
    @staticmethod
    def read_ndarray(file):
        try:
            data = np.load(file,  allow_pickle=False)
        except ValueError:
            data = np.load(file,  allow_pickle=True)
        return data

    @ staticmethod
    def get_upper_tri_mat(data, one_file_per_modality):
        """Get upper triangular matrix

        If the matrix is symmetric, then I extract the upper triangular matrix.

        Parameters:
        ----------
        data: list of ndarray or DataFrame
        
        one_file_per_modality: bool
            If one file per modality, which used for generating mask.

        Return:
        -------
        data_: list of ndarray or DataFrame
            data that the upper triangular matrix was extracted or the original data.
            
        mask: ndarray
            Internal mask of data (mask sure that all item in data have the same dimension and type) 
        """

        # If data is symmetric matrix, then only extract upper triangule matrix
        data_ = []
        for dd in data:
            dd_ = dd.copy()
            if ((len(dd.shape) == 2) and (dd.shape[0] == dd.shape[1])):
                dd_[np.eye(dd.shape[0]) == 1] = 0
                if np.all(np.abs(dd_-dd_.T) < 1e-8):  
                    mask = np.triu(np.ones(dd.shape),1) == 1
                    data_.append(dd[mask])
                else:
                    mask = np.ones(np.shape(dd)) if not one_file_per_modality else np.ones(np.shape(dd)[1]) 
                    data_.append(dd)
            else:
                mask = np.ones(np.shape(dd)) if not one_file_per_modality else np.ones(np.shape(dd)[1])
                data_.append(dd)

        return data_, mask == 1
            

    @staticmethod
    def extract_id(files):
        """Extract subject unique ID from file names

        Parameters:
        ----------
        files : list of file paths

        Returns:
        -------
        unique_identifier: pd.DataFrame
            Subjects' name or unique idenfity
        """
        
        unique_identifier = [os.path.basename(file).split(".")[0] for file in files]
        unique_identifier = [re.findall(r'.*(sub.?[0-9]+).*', name)[0] if re.findall(r'.*(sub.?[0-9]+).*', name) != [] else "" for name in unique_identifier]
        unique_identifier = pd.DataFrame(unique_identifier)
        unique_identifier.columns = ["__ID__"]
        return unique_identifier

    