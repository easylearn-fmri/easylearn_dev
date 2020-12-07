#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
This class is the base class for regression
Author: Mengshi dong <dongmengshi1990@163.com>
"""

import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import pearsonr

from eslearn.machine_learning.base import AbstractSupervisedMachineLearningBase
from eslearn.utils.timer import timer

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


class BaseRegression():
    """Base class for regression

    Parameters
    ----------
    None

    Attributes
    ----------
    model_: Fited model object, default None

    weights_: ndarray of shape(n_class, n_features) if the model is linear model, else shape(1,n_features), default None
        Feature weights of the fited model

    weights_norm_: ndarray of shape(n_class, n_features) if the model is linear model, else shape(1,n_features), default None
        Normalized feature weights. Using StandardScaler (z-score) to get the normalized feature weights.

    """

    def __init__(self, 
                metric=mean_squared_error,
                search_strategy='random', 
                k=5, 
                n_iter_of_randomedsearch=10, 
                n_jobs=1, 
                location='cachedir',
                verbose=False):
        
        self.metric = metric
        self.model_ = None
        weights_ = None
        weights_norm_ = None

        self.search_strategy = search_strategy
        self.k = k
        self.n_iter_of_randomedsearch = n_iter_of_randomedsearch
        self.n_jobs = n_jobs
        self.location = location
        self.verbose = verbose
        
    @timer 
    def fit_(self, model, x=None, y=None):
        """Fit the scikit-learn search or pipeline model
        """
        
        model.fit(x, y)
        # Delete the temporary cache before exiting
        self.memory.clear(warn=False)
        return self

    def predict_(self, x):
        y_hat = self.model_.predict(x)
        return y_hat

    def get_weights_(self, x=None, y=None):
        """
        If the model is linear model, the weights are coefficients.
        If the model is not the linear model, the weights are calculated by occlusion test <Transfer learning improves resting-state functional
        connectivity pattern analysis using convolutional neural networks>.
        """
        
        if self.is_search:
            best_model = self.model_.best_estimator_
        else:
            best_model = self.model_
            
        feature_preprocessing = best_model['feature_preprocessing']
        dim_reduction = best_model.get_params().get('dim_reduction',None)
        feature_selection =  best_model.get_params().get('feature_selection', None)
        estimator =  best_model['estimator']

        # Get weight according to model type: linear model or nonlinear model
        if hasattr(estimator, "coef_"):
            coef =  estimator.coef_.reshape(1,-1)                
            if feature_selection and (feature_selection != "passthrough"):
                weights_ = feature_selection.inverse_transform(coef)
            else:
                weights_ = coef

            if dim_reduction and (dim_reduction != "passthrough"):
                weights_ = dim_reduction.inverse_transform(weights_)
            
        else:  # Nonlinear model
        # TODO: Consider the problem of slow speed caused by a large number of features
            x_reduced_selected = x.copy()
            if feature_preprocessing and (feature_preprocessing != "passthrough"):
                x_reduced_selected = feature_preprocessing.fit_transform(x_reduced_selected)
            if dim_reduction and (dim_reduction != "passthrough"):
                x_reduced_selected = dim_reduction.fit_transform(x_reduced_selected)
            if feature_selection and (feature_selection != "passthrough"):
                x_reduced_selected = feature_selection.fit_transform(x_reduced_selected, y)

            y_hat = self.model_.predict(x)
            score_true = self.metric(y, y_hat)
            len_feature = np.shape(x_reduced_selected)[1]
            
            if len_feature > 1000:
                 print("***There are {len_feature} features, it may take a long time to get the weight!***\n")
                 print("***I suggest that you reduce the dimension of features***\n")
            
            weights_ = np.zeros([1,len_feature])
            for ifeature in range(len_feature):
                print(f"Getting weight for the {ifeature+1}th feature...\n")
                x_ = np.array(x_reduced_selected).copy()
                x_[:,ifeature] = 0
                y_hat = estimator.predict(x_)
                weights_[0, ifeature] = score_true - self.metric(y, y_hat)
            
            # Back to original space
            weights_ = np.reshape(weights_, [1, -1])
            if feature_selection and (feature_selection != "passthrough"):
                weights_ = feature_selection.inverse_transform(weights_)
            if dim_reduction and (dim_reduction != "passthrough"):
                weights_  = dim_reduction.inverse_transform(weights_)            
                
        # Normalize weights using z-score method
        weights_norm_ = StandardScaler().fit_transform(weights_.T).T

        return weights_, weights_norm_


class StatisticalAnalysis(BaseRegression):
    """Statistical analysis for regression results

    Parameters:
    ----------
    method_statistical_analysis: str, "Binomial test" or "Permutation test"
        Method of statistical analysis

    sorted_targets: ndarray with shape of [n_samples,]
        Sorted targets during cross-validation training, which is used to perform binomial test.

    predict_targets: ndarray with shape of [n_samples,]
       Predicted label of samples

    model: object
        Machine learning model with fit method, which is used to perform permutation test.

    feature: ndarray with shape of [n_samples, n_features]
        Features of samples.
   
    label: ndarray with shape of [n_samples]
        Real label of samples with original order.

    prep_: object
        Preprocessing object with such fit_transform, transform methods. prep_ is derived from training stage.

    time_permutation: int
        How many times to permute training labels.

    method_model_evaluation: sklearn object's instance
        Method of model evaluation, e.g. StratifiedKFold()

    out_dir: path str
        Output directory used to save results.
    """

    def __init__(self, method_statistical_analysis,
        sorted_targets=None, predict_targets=None,
        model=None, features=None, targets=None, 
        prep_=None, time_permutation=None, method_model_evaluation=None,
        out_dir=None):

        super().__init__()
        self.method_statistical_analysis = method_statistical_analysis
        self.sorted_targets = sorted_targets
        self.predict_targets = predict_targets
        self.model = model
        self.features = features
        self.targets = targets
        self.prep_ = prep_
        self.time_permutation = time_permutation
        self.method_model_evaluation = method_model_evaluation
        self.out_dir = out_dir

    def fit(self):
        """Statistical analysis"""

        print("Statistical analysis...\n")
        if self.method_statistical_analysis == "Binomial test":
            pvalue_metric = self.pearson_test()
            permuted_score = None
        elif self.method_statistical_analysis == "Permutation test":
            pvalue_metric, permuted_score = self.permutation_test()

        # Save outputs
        self.outputs = {}
        self.outputs.update({"pvalue_metric": pvalue_metric, "permutated_score": permuted_score})
        pickle.dump(self.outputs, open(os.path.join(self.out_dir, "stat.pickle"), "wb"))
        return self

    def pearson_test(self):
        r, pvalue_metric = pearsonr(np.array(self.sorted_targets), np.array(self.predict_targets))
        print(f"p value for metric = {pvalue_metric:.3f}")
        return pvalue_metric

    def permutation_test(self):
        print(f"Permutation test: {self.time_permutation} times...\n")
                
        permuted_score = []

        for i in range(self.time_permutation):
            print(f"{i+1}/{self.time_permutation}...\n")      
            permuted_score = []
            for train_index, test_index in self.method_model_evaluation.split(self.features, self.targets):
                feature_train = self.features[train_index, :]
                feature_test = self.features[test_index, :]
                
                # Preprocessing
                feature_train = self.prep_.fit_transform(feature_train)
                feature_test = self.prep_.transform(feature_test)
                
                permuted_target_train = self.targets[train_index][np.random.permutation(len(train_index))]
                # target_train = self.targets[train_index]
                target_test = self.targets[test_index]

                # Fit
                self.fit_(self.model, feature_train, permuted_target_train)

                # Predict
                y_prob = self.predict_(feature_test)
                
                # Eval performances
                score = self.metric(target_test, y_prob)  
                permuted_score.append(score)
             
            # Average performances of one permutation
            permuted_score.append(np.mean(permuted_score))

        # Get p values
        pvalue_metric = self.calc_pvalue(permuted_score, np.mean(self.real_score))
        print(f"p value for metric = {pvalue_metric:.3f}")
        return pvalue_metric, permuted_score

    @staticmethod
    def calc_pvalue(permuted_performance, real_performance):
        return (np.sum(np.array(permuted_performance) <= np.array(real_performance)) + 1) / (len(permuted_performance) + 1)


if __name__ == "__main__":
    basereg = BaseRegression()

