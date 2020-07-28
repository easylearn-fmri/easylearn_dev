#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
This class is the base class for classification
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, max_error
from sklearn.pipeline import Pipeline
from joblib import Memory
from shutil import rmtree
from abc import abstractmethod, ABCMeta
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


class BaseClassification(metaclass=ABCMeta):
    """Base class for classification"""

    def __init__(self):
        self.model = None
        self.weights_ = None
        self.weights_norm_ = None

    def get_weights_(self, x=None, y=None):
        """
        If the model is linear model, the weights are coefficients.
        If the model is not the linear model, the weights are calculated by occlusion test <Transfer learning improves resting-state functional
        connectivity pattern analysis using convolutional neural networks>.
        """
        
        best_model = self.model.best_estimator_
        estimator =  best_model['estimator']
        dim_reduction = best_model.get_params().get('dim_reduction',None)
        feature_selection =  best_model.get_params().get('feature_selection', None)

        # Get weight according to model type: linear model or nonlinear model
        if hasattr(estimator, "coef_"):
            coef =  estimator.coef_
            if feature_selection and (feature_selection != "passthrough"):
                self.weights_ = [np.zeros(np.size(feature_selection.get_support())) for i in range(len(coef))]
            else:
                self.weights_ = [[] for i in range(len(coef))]
                
            for i, coef_ in enumerate(coef):
                if feature_selection and (feature_selection != "passthrough"):
                    self.weights_[i][feature_selection.get_support()] = coef_
                else:
                    self.weights_[i] = coef_

                if dim_reduction and (dim_reduction != "passthrough"):
                    self.weights_[i] = dim_reduction.inverse_transform(self.weights_[i])
        else:  # Nonlinear model
            self.weights_ = []
            y_hat = self.model.predict(x)
            score_true = self.metric(y, y_hat)
            len_feature = np.shape(x)[1]
            for ifeature in range(len_feature):
                x_ = np.array(x).copy()
                x_[:,ifeature] = 0
                y_hat = self.model.predict(x_)
                self.weights_.append(score_true-self.metric(y, y_hat))
                
        # Normalize weights
        self.weights_norm_ = [wei/np.sum(np.power(np.e,wei)) for wei in self.weights_]
        
    
class PipelineSearch_(BaseClassification):
    """Make pipeline_"""

    def __init__(self, 
                 search_strategy='random', 
                 k=5, metric=max_error, 
                 n_iter_of_randomedsearch=10, 
                 n_jobs=1, 
                 location='cachedir',
                 verbose=False
    ):

        super().__init__()
        self.search_strategy = search_strategy
        self.k = k
        self.metric = metric
        self.n_iter_of_randomedsearch = n_iter_of_randomedsearch
        self.n_jobs = n_jobs
        self.location = location
        self.verbose = verbose
        
        self.pipeline_ = None
        self.param_search_ = None

    def make_pipeline_(self, 
                        method_feature_preprocessing=None, 
                        param_feature_preprocessing=None,
                        method_dim_reduction=None, 
                        param_dim_reduction=None, 
                        method_feature_selection=None, 
                        param_feature_selection=None, 
                        method_machine_learning=None, 
                        param_machine_learning=None):
        
        """Construct pipeline_

        Currently, the pipeline_ only supports one specific method for corresponding method, 
        e.g., only supports one dimension reduction method for dimension reduction.
        In the next version, the pipeline_ will support multiple methods for each corresponding method.

        Parameters:
        ----------
            method_feature_preprocessing: [list of] sklearn module, such as [PCA()]
                    method of dimension reduction.

            param_feature_preprocessing: dictionary [or list of dictionaries], {'reduce_dim__n_components':[0.5,0.9]}, 
                parameters of dimension reduction, such as components of PCA.

            method_dim_reduction: [list of] sklearn module, such as [PCA()]
                method of dimension reduction.

            param_dim_reduction: dictionary [or list of dictionaries], {'reduce_dim__n_components':[0.5,0.9]}, 
                parameters of dimension reduction, such as components of PCA.

            method_feature_selection: [list of] sklearn module, such as [LinearSVC()]
                method of feature selection.

            param_feature_selection: dictionary [or list of dictionaries], {'feature_selection__k': [0.5,0.9]},
                parameters of feature selection, such as How many features to be kept.

            method_machine_learning: [list of] sklearn module, such as [LinearSVC()]
                method of feature selection.

            param_machine_learning: dictionary [or list of dictionaries], such as 
            {'estimator__penalty': ['l1', 'l2'], 'estimator__C': [10]}
                parameters of feature selection.

        """

        
        self.memory = Memory(location=self.location, verbose=self.verbose)

        self.pipeline_ = Pipeline(steps=[
                ('feature_preprocessing','passthrough'),
                ('dim_reduction', 'passthrough'),
                ('feature_selection', 'passthrough'),
                ('estimator', 'passthrough'),
            ], 
            memory=self.memory
        )

        # Set parameters of gridCV
        print("Setting parameters of gridCV...\n")
        
        self.param_search_ = {}

        if method_feature_preprocessing:
            self.param_search_.update({'feature_preprocessing':method_feature_preprocessing})
        if param_feature_preprocessing:          
            self.param_search_.update(param_feature_preprocessing)
            
        if method_dim_reduction:
            self.param_search_.update({'dim_reduction':method_dim_reduction})
        if param_dim_reduction:
            self.param_search_.update(param_dim_reduction)
            # Get number of features that pass the dimension reduction
            # pca = method_dim_reduction[0]
            # xx = pca.fit_transform(x)
                
        if method_feature_selection:
            self.param_search_.update({'feature_selection': method_feature_selection})
        if param_feature_selection:
            self.param_search_.update(param_feature_selection)
            
        if method_machine_learning:
            self.param_search_.update({'estimator': method_machine_learning})
        if param_machine_learning:
            self.param_search_.update(param_machine_learning)
            

        return self
    
    def fit_pipeline_(self, x=None, y=None):
        """Fit the pipeline_"""
        
        # TODO: Extending to other CV methods
        cv = StratifiedKFold(n_splits=self.k)  # Default is StratifiedKFold
        if self.search_strategy == 'grid':
            self.model = GridSearchCV(
                self.pipeline_, n_jobs=self.n_jobs, param_grid=self.param_search_, cv=cv, 
                scoring = make_scorer(self.metric), refit=True
            )
            # print(f"GridSearchCV fitting (about {iteration_num} times iteration)...\n")

        elif self.search_strategy == 'random':
            self.model = RandomizedSearchCV(
                self.pipeline_, n_jobs=self.n_jobs, param_distributions=self.param_search_, cv=cv, 
                scoring = make_scorer(self.metric), refit=True, n_iter=self.n_iter_of_randomedsearch,
            )
        
            # print(f"RandomizedSearchCV fitting (about {iteration_num} times iteration)...\n")
        else:
            print("Please specify which search strategy!\n")
            return

        self.model.fit(x, y)

        # Delete the temporary cache before exiting
        self.memory.clear(warn=False)
        rmtree(self.location)
        return self

    def predict(self, x):
        y_hat = self.model.predict(x)
        
        # TODO?
        if hasattr(self.model, 'decision_function'):
            y_prob = self.model.decision_function(x)
        elif hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(x)[:,1]
        else:
            y_prob = y_hat
                
        return y_hat, y_prob

