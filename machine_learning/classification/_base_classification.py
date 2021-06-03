#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
This class is the base class for classification
"""

import numpy as np
import os
import pickle
import time
from progressbar import Percentage, ProgressBar, Bar, Timer, ETA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from joblib import Memory
from shutil import rmtree
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, auc, f1_score

from eslearn.machine_learning.base import AbstractSupervisedMachineLearningBase
from eslearn.utils.timer import  timer
from eslearn.statistical_analysis import el_binomialtest
from eslearn.model_evaluator import ModelEvaluator


warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


class BaseClassification():
    """Base class for classification

    Parameters
    ----------
    None

    Attributes
    ----------
    model_: Fited model object, default None

    weights_: ndarray of shape(n_class, n_feature) if the model is linear model, else shape(1,n_feature), default None
        Feature weights of the fited model

    weights_norm_: ndarray of shape(n_class, n_feature) if the model is linear model, else shape(1,n_feature), default None
        Normalized feature weights. Using StandardScaler (z-score) to get the normalized feature weights.

    """

    def __init__(self,
                search_strategy='grid', 
                gridcv_k=3, 
                metric=accuracy_score, 
                n_iter_of_randomedsearch=10, 
                n_jobs=2):

        self.search_strategy = search_strategy
        self.gridcv_k = gridcv_k
        self.metric = metric
        self.n_iter_of_randomedsearch = n_iter_of_randomedsearch
        self.n_jobs = n_jobs
        
        self.model_ = None
        weights_ = None
        weights_norm_ = None

    # @timer 
    def fit_(self, model, x=None, y=None, memory=None):
        """Fit the scikit-learn search or pipeline model
        """
        
        model.fit(x, y)
        # Delete the temporary cache before exiting
        if memory is not None: memory.clear(warn=False)
        return self
    
    def predict_(self, model, x):
        # TODO: extend this to multiple-class classification

        y_pred = model.predict(x)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(x)
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(x)
        else:
            y_prob = y_pred
                
        return y_pred, y_prob
    
    def get_weights_(self, x=None, y=None):
        """Get model weights

        Parameters:
        ----------
        x : array_like with shape [n_samples, n_feature]
            Features

        y: array_like with shape [n_samples,]
            Targets

        Returns:
        -------
        weights_: ndarray with shape [n_feature,]
            Model weights

        weights_norm_: ndarray with shape [n_feature,]
            Normalized model weights

        Notes:
        -----
        If the model is linear model, the weights are coefficients.
        If the model is not the linear model, the weights are calculated by occlusion test 
        <Transfer learning improves resting-state functional
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
        if hasattr(estimator, "coef_"):  # Linear model
            coef =  estimator.coef_
            if feature_selection and (feature_selection != "passthrough"):
                weights_ = feature_selection.inverse_transform(coef)
            else:
                weights_ = coef
                
            if dim_reduction and (dim_reduction != "passthrough"):
                weights_ = dim_reduction.inverse_transform(weights_)
        
        else:  # Nonlinear model
        # TODO: Consider the problem of slow speed caused by a large number of feature
            x_reduced_selected = x.copy()
            if feature_preprocessing and (feature_preprocessing != "passthrough"):
                x_reduced_selected = feature_preprocessing.fit_transform(x_reduced_selected)
            if dim_reduction and (dim_reduction != "passthrough"):
                x_reduced_selected = dim_reduction.fit_transform(x_reduced_selected)
            if feature_selection and (feature_selection != "passthrough"):
                x_reduced_selected = feature_selection.fit_transform(x_reduced_selected, y)
            
            y_pred = self.model_.predict(x)
            score_true = self.metric(y, y_pred)
            len_feature = x_reduced_selected.shape[1]
            weights_ = np.zeros([1,len_feature])
            
            if len_feature > 1000:
                 print(f"***There are {len_feature} feature, it may take a long time to get the weight!***\n")
                 print("***I suggest that you reduce the dimension of feature***\n")
                 
            for ifeature in range(len_feature):
                print(f"Getting weight for the {ifeature+1}th feature...\n")
                x_ = x_reduced_selected.copy()
                x_[:,ifeature] = 0
                y_pred = estimator.predict(x_)
                weights_[0, ifeature] = score_true - self.metric(y, y_pred)
            
            # Back to original space
            if feature_selection and (feature_selection != "passthrough"):
                weights_ = feature_selection.inverse_transform(weights_)
            if dim_reduction and (dim_reduction != "passthrough"):
                weights_  = dim_reduction.inverse_transform(weights_)            
                
        # Normalize weights
        weights_norm_ = StandardScaler().fit_transform(weights_.T).T

        return weights_, weights_norm_


class StatisticalAnalysis(BaseClassification):
    """Statistical analysis for classification results

    Parameters:
    ----------
    method_statistical_analysis: str, "Binomial test" or "Permutation test"
        Method of statistical analysis

    sorted_label: ndarray with shape of [n_samples,]
        Sorted real label during cross-validation training, which is used to perform binomial test.

    predict_label: ndarray with shape of [n_samples,]
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

    def __init__(self, 
        method_statistical_analysis=None,
        sorted_label=None, 
        predict_label=None,
        model=None, 
        feature=None, 
        label=None, 
        prep_=None, 
        time_permutation=None, 
        method_unbalance_treatment_=None,
        method_model_evaluation=None,
        real_accuracy=None,
        real_sensitivity=None,
        real_specificity=None,
        real_auc=None,
        memory=None,
        out_dir=None):

        super().__init__()
        self.method_statistical_analysis = method_statistical_analysis
        self.sorted_label = sorted_label
        self.predict_label = predict_label
        self.model = model
        self.feature = feature
        self.label = label
        self.prep_ = prep_
        self.time_permutation = time_permutation
        self.method_unbalance_treatment_ = method_unbalance_treatment_
        self.method_model_evaluation = method_model_evaluation
        self.real_accuracy = real_accuracy
        self.real_sensitivity = real_sensitivity
        self.real_specificity = real_specificity
        self.real_auc = real_auc
        self.memory = memory
        self.out_dir = out_dir

    def fit(self):
        """Statistical analysis"""

        print("Statistical analysis...\n")
        if self.method_statistical_analysis == "Binomial/Pearson-R test":
            pvalue_acc, pvalue_sens, pvalue_spec, pvalue_auc = self.binomial_test()
            permuted_score = None
        elif self.method_statistical_analysis == "Permutation test":
            pvalue_acc, pvalue_sens, pvalue_spec, pvalue_auc = self.permutation_test()

        # Save outputs
        self.outputs = {}
        self.outputs.update({"pvalue_acc": pvalue_acc, "pvalue_sens": pvalue_sens, 
                            "pvalue_spec": pvalue_spec, "pvalue_auc": pvalue_auc})
        pickle.dump(self.outputs, open(os.path.join(self.out_dir, "stat.pickle"), "wb"))
        return self

    def binomial_test(self):
        k = np.sum(np.array(self.sorted_label) - np.array(self.predict_label)==0)
        n = len(self.sorted_label)
        pvalue_acc = el_binomialtest.binomialtest(n, k, 0.5)
        pvalue_sens = None
        pvalue_spec = None
        pvalue_auc = None
        print(f"p value for acc = {pvalue_acc:.3f}")
        el_binomialtest.lc_plot(n, k, 0.5, "Binomial test", os.path.join(self.out_dir, "binomial_test.pdf"))
        return pvalue_acc, pvalue_sens, pvalue_spec, pvalue_auc

    def permutation_test(self):
        print(f"Permutation test: {self.time_permutation} times...\n")
                
        self.permuted_accuracy = []
        self.permuted_sensitivity = []
        self.permuted_specificity = []
        self.permuted_auc = []
        count = 0
        widgets = ['Permutation testing', Percentage(), ' ', Bar('='),' ', Timer(),  ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=self.time_permutation).start()
        
        for i in range(self.time_permutation):
            # Get training and test datasets         
            accuracy = []
            sensitivity = []
            specificity = []
            AUC = []
            for train_index, test_index in self.method_model_evaluation.split(self.feature, self.label):
                feature_train = self.feature[train_index, :]
                feature_test = self.feature[test_index, :]
                permuted_target_train = self.label[train_index][np.random.permutation(len(train_index))]
                
                target_test = self.label[test_index]

                # Preprocessing
                feature_train = self.prep_.fit_transform(feature_train)
                feature_test = self.prep_.transform(feature_test)

                # Resample
                imbalance_resample = self.method_unbalance_treatment_
                if imbalance_resample:
                    feature_train, permuted_target_train = imbalance_resample.fit_resample(feature_train, permuted_target_train)

                # Fit
                self.fit_(self.model, feature_train, permuted_target_train, self.memory)
                
                # Predict
                y_pred, y_prob = self.predict_(self.model, feature_test)
                
                # Eval performances
                acc, sens, spec, auc_, _ = ModelEvaluator().binary_evaluator(
                    target_test, y_pred, y_prob,
                    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                    verbose=False, is_showfig=False, is_savefig=False
                )
                
                accuracy.append(acc)
                sensitivity.append(sens)
                specificity.append(spec)
                AUC.append(auc_)
            
            # Average performances of one permutation
            self.permuted_accuracy.append(np.mean(accuracy))
            self.permuted_sensitivity.append(np.mean(sensitivity))
            self.permuted_specificity.append(np.mean(specificity))
            self.permuted_auc.append(np.mean(AUC))
            
            # Progress bar
            pbar.update(count)
            count += 1
        
        pbar.finish()

        # Get p values
        pvalue_acc = self.calc_pvalue(self.permuted_accuracy, np.mean(self.real_accuracy))
        pvalue_sens = self.calc_pvalue(self.permuted_sensitivity, np.mean(self.real_sensitivity))
        pvalue_spec = self.calc_pvalue(self.permuted_specificity, np.mean(self.real_specificity))
        pvalue_auc = self.calc_pvalue(self.permuted_auc, np.mean(self.real_auc))
        
        print(f"p value for acc = {pvalue_acc:.3f}")
        return pvalue_acc, pvalue_sens, pvalue_spec, pvalue_auc

    @staticmethod
    def calc_pvalue(permuted_performance, real_performance):
        return (np.sum(np.array(permuted_performance) >= np.array(real_performance)) + 1) / (len(permuted_performance) + 1)

   
        
if __name__=="__main__":
    baseclf = BaseClassification()