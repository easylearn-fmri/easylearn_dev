#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Author: Mengshi Dong <dongmengshi1990@163.com>
"""

import time
import os
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, max_error

from eslearn.base import BaseMachineLearning, DataLoader
from eslearn.machine_learning.regression._base_regression import BaseRegression
from eslearn.model_evaluator import ModelEvaluator


x, y = datasets.make_regression(n_samples=200, n_informative=50, n_features=100, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


class Regression(BaseMachineLearning, DataLoader, BaseRegression):
    
    def __init__(self, configuration_file, out_dir):
        BaseMachineLearning.__init__(self, configuration_file)
        DataLoader.__init__(self, configuration_file)
        BaseRegression.__init__(self)
        self.metric = mean_absolute_error
        self.out_dir = out_dir

    def main_run(self):
        
        # Get all inputs
        self.load_data()
        self.get_all_inputs()

        # Make pipeline
        self.make_pipeline_()
        
       # Get training and test datasets        
        cv = self.method_model_evaluation_ 
        self.target_test_all = []
        self.pred_prob = []
        self.real_score = []
        weights = []
        subname = []
        for train_index, test_index in cv.split(self.features_, self.targets_):
            feature_train = self.features_[train_index, :]
            feature_test = self.features_[test_index, :]
            target_train = self.targets_[train_index]
            target_test = self.targets_[test_index]
            self.target_test_all.extend(target_test)

            subname_ = self.id_[test_index]
            subname.extend(subname_)

            # Resample
            imbalance_resample = self.method_unbalance_treatment_
            feature_train, target_train = imbalance_resample.fit_resample(feature_train, target_train)
            print(f"After re-sampling, the sample size are: {sorted(Counter(target_train).items())}")
            
            # Fit
            self.fit_(feature_train, target_train)
            self.get_weights_(feature_train, target_train)

            # Predict
            y_prob = self.predict_(feature_test)
            
            # Eval performances
            score = self.metric(target_test, y_prob)  
            self.real_score.append(score)
            self.pred_prob.extend(y_prob)

            weights.append(self.weights_)          
        

        # Eval performances for all fold
        # out_name_perf = os.path.join(self.out_dir, "classification_performances.pdf")
        # all_score = ModelEvaluator().regression_evaluator(
        #     self.target_test_all, self.pred_prob,
        #     accuracy_kfold=self.real_score, 
        #     verbose=1, is_showfig=True, is_savefig=True, legend1='Controls', legend2='Patients', out_name=out_name_perf
        # )

        # Save weight
        self.save_weight(weights, self.out_dir)

        # # Statistical analysis
        # print("Statistical analysis...\n")
        # self.run_statistical_analysis()
        
        # # Save outputs
        # outputs = { "subname": subname, "test_targets": self.target_test_all, "test_probability": self.pred_prob, 
        #             "accuracy": self.real_score, "pvalue_spec": self.pvalue_score, 
        # }

        # pickle.dump(outputs, open(os.path.join(self.out_dir, "outputs.pickle"), "wb"))


        return self
    
    def run_statistical_analysis(self):
        type_dict = {"Binomial test":self.binomial_test, "Permutation test":self.permutation_test}
        type_dict[self.method_statistical_analysis_]()

    # TODO: change it to calculate P value of Pearson's correlation coefficient
    def binomial_test(self):
        k = np.sum(np.array(self.target_test_all) - np.array(self.pred_label)==0)
        n = len(self.target_test_all)
        self.pvalue_acc, sum_prob, prob, randk = el_binomialtest.binomialtest(n, k, 0.5, 0.5)
        self.pvalue_auc = None
        self.pvalue_sens = None
        self.pvalue_spec = None
        print(f"p value for acc = {self.pvalue_acc:.3f}")
        return self

    def permutation_test(self):
        print(f"Permutation test: {self.param_statistical_analysis_} times...\n")
        
        self.preprocessing()
        
        self.permuted_score = []

        for i in range(self.param_statistical_analysis_):
            print(f"{i+1}/{self.param_statistical_analysis_}...\n")
            # Get training and test datasets         
            
            permuted_score = []
            self.target_test_all = []
            for train_index, test_index in cv.split(self.features_, self.targets_):
                feature_train = self.features_[train_index, :]
                feature_test = self.features_[test_index, :]
                permuted_target_train = self.targets_[train_index][np.random.permutation(len(train_index))]
                target_train = self.targets_[train_index]
                target_test = self.targets_[test_index]
                self.target_test_all.extend(target_test)

                subname_ = self.id_[test_index]
                subname.extend(subname_)

                # Resample
                imbalance_resample = self.method_unbalance_treatment_
                permuted_target_train, target_train = imbalance_resample.fit_resample(permuted_target_train, target_train)
                print(f"After re-sampling, the sample size are: {sorted(Counter(target_train).items())}")
                
                # Fit
                self.fit_(permuted_target_train, target_train)
                self.get_weights_(permuted_target_train, target_train)

                # Predict
                y_prob = self.predict_(feature_test)
                
                # Eval performances
                score = self.metric(target_test, y_prob)  
                permuted_score.append(score)
             
            # Average performances of one permutation
            self.permuted_score.append(np.mean(permuted_score))

        # Get p values
        self.pvalue_acc = self.calc_pvalue(self.permuted_score, np.mean(self.real_score))
        print(f"p value for acc = {self.pvalue_acc:.3f}")
        return self

    @staticmethod
    def calc_pvalue(permuted_performance, real_performance):
        return (np.sum(np.array(permuted_performance) <= np.array(real_performance)) + 1) / (len(permuted_performance) + 1)
    

if __name__ == "__main__":
    time_start = time.time()
    clf = Regression(configuration_file=r'D:\My_Codes\easylearn\eslearn\GUI\test\configuration_file_reg.json',
                     out_dir=r"D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\GUI\tests")
    clf.main_run()
    time_end = time.time()
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)