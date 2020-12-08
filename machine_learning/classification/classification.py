#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import nibabel as nib

from eslearn.base import BaseMachineLearning, DataLoader
from eslearn.preprocessing.preprocessing import Denan
from eslearn.machine_learning.classification._base_classification import BaseClassification
from eslearn.machine_learning.classification._base_classification import StatisticalAnalysis
from eslearn.model_evaluator import ModelEvaluator
from eslearn.statistical_analysis import el_binomialtest


class Classification(BaseMachineLearning, DataLoader, BaseClassification):
    
    def __init__(self, configuration_file, out_dir):
        BaseMachineLearning.__init__(self, configuration_file)
        DataLoader.__init__(self, configuration_file)
        BaseClassification.__init__(self)  # StatisticalAnalysis inherit from BaseClassification
        self.out_dir = out_dir

    def preprocessing(self):
        # Get all inputs
        self.load_data()
        self.get_all_inputs()
        # Make pipeline
        self.make_sklearn_search_model_()

    def main_run(self):
        self.preprocessing()

        # Get training and test datasets         
        self.real_accuracy = []
        self.real_sensitivity = []
        self.real_specificity = []
        self.real_auc = []
        self.pred_label = []
        pred_prob = []
        weights = []
        self.target_test_all = []
        subname = []
        for train_index, test_index in self.method_model_evaluation_.split(self.features_, self.targets_):
            feature_train = self.features_[train_index, :]
            feature_test = self.features_[test_index, :]
            target_train = self.targets_[train_index]
            target_test = self.targets_[test_index]

            subname_ = self.id_[test_index]
            subname.extend(subname_)
            
            # Preprocessing
            self.prep_ = Denan(how='median')
            feature_train = self.prep_.fit_transform(feature_train)
            feature_test = self.prep_.transform(feature_test)

            self.target_test_all.extend(target_test)

            # Resample
            imbalance_resample = self.method_unbalance_treatment_
            if imbalance_resample:
                print(f"Before re-sampling, the sample size are: {sorted(Counter(target_train).items())}")
                feature_train, target_train = imbalance_resample.fit_resample(feature_train, target_train)
                print(f"After re-sampling, the sample size are: {sorted(Counter(target_train).items())}")
            
            # Fit
            self.fit_(self.model_, feature_train, target_train, self.memory)
            
            # Weights
            weights_, _ = self.get_weights_(feature_train, target_train)
            
            # Predict
            y_pred, y_prob = self.predict_(self.model_, feature_test)
            
            # Eval performances
            acc, sens, spec, auc_, _ = ModelEvaluator().binary_evaluator(
                target_test, y_pred, y_prob,
                accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                verbose=False, is_showfig=False, is_savefig=False
            )
            
            self.real_accuracy.append(acc)
            self.real_sensitivity.append(sens)
            self.real_specificity.append(spec)
            self.real_auc.append(auc_)
            self.pred_label.extend(y_pred)
            pred_prob.extend(y_prob)
            weights.append(weights_)
        
        
        # Save weight
        self.save_weight(weights, self.out_dir)
        
        # Eval performances for all fold
        out_name_perf = os.path.join(self.out_dir, "classification_performances.pdf")
        acc, sens, spec, auc, _ = ModelEvaluator().binary_evaluator(
            self.target_test_all, self.pred_label, pred_prob,
            accuracy_kfold=self.real_accuracy, 
            sensitivity_kfold=self.real_sensitivity, 
            specificity_kfold=self.real_specificity, 
            AUC_kfold=self.real_auc,
            verbose=1, is_showfig=True, is_savefig=True, legend1='Controls', legend2='Patients', out_name=out_name_perf
        )

        
        # Save outputs
        self.outputs = { 
            "preprocessor": self.prep_, "model":self.model_, 
            "subname": subname, "test_targets": self.target_test_all, "test_prediction": self.pred_label, 
            "test_probability": pred_prob, "accuracy": self.real_accuracy,
            "sensitivity": self.real_sensitivity, "specificity":self.real_specificity, "auc": self.real_auc
        }

        pickle.dump(self.outputs, open(os.path.join(self.out_dir, "outputs.pickle"), "wb"))
        
        return self

    def run_statistical_analysis(self):
        stat = StatisticalAnalysis(self.method_statistical_analysis_,
            self.target_test_all, 
            self.pred_label,
            self.model_, 
            self.features_, 
            self.targets_, 
            self.prep_, 
            self.param_statistical_analysis_, 
            self.method_unbalance_treatment_, 
            self.method_model_evaluation_,
            self.real_accuracy,
            self.real_sensitivity,
            self.real_specificity ,
            self.real_auc,
            self.memory,
            self.out_dir
        )

        stat.fit()

if __name__ == "__main__":
    time_start = time.time()
    clf = Classification(configuration_file=r'D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\machine_learning\classification\tests\clf_configuration.json', 
                         out_dir=r"D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\machine_learning\classification\tests") 
    clf.main_run()
    time_end = time.time()
    # print(clf.param_search_)
    # print(clf.pipeline_)
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)
    
    clf.run_statistical_analysis()
    
    