# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 21:55:50 2020
This script is designed to predict unseen data using pre-trained model(s).
If have multiple models, eslearn uses the mean prediction as final prediction.
@author: Li Chao, Dong Mengshi
"""

import numpy as np
import pandas as pd
import pickle
from eslearn.base import BaseMachineLearning, DataLoader


class Predict():

    def __init__(self, data_file=None, model_file=None):
        """
        Parameters:
        ----------
        data_file: file path
            Where is the data loading json file

        model_file: file path
            Where is the model file (outputs.pickle)
        
        Returns:
        -------
        predict_proba: ndarray with shape of [n_samples, [*n_classes]]
            predict probability

        predict_label: ndarray with shape of [n_samples, ]
            predict label
        """

        self.data_file = data_file
        self.model_file = model_file

    def run(self):
        """ 
        """

        # Load model
        self.model_file = r"F:\线上讲座\demo_data\szVShc_fc\outputs.pickle"
        output = pickle.load(open(self.model_file, "rb"))
        preprocessor = output["preprocessor"]
        best_model = output["model"]
        if hasattr(best_model, "best_estimator_"):
            best_model = best_model.best_estimator_

        # Get data
        self.data_file = './dataLoadingTest.json'
        data_loader = DataLoader(configuration_file=self.data_file)
        data_loader.load_data()
        feature = data_loader.features_  
        target = data_loader.targets_ 
        
        # Predict
        if not isinstance(best_model, list):  # Force the model and preprocessor is a list
            best_model_ = [best_model, best_model]
            
        if not isinstance(preprocessor, list):  # Force the model and preprocessor is a list
            preprocessor_ = [preprocessor, preprocessor]
        
        predict_label = []
        pred_prob = []
        for prep, model_ in zip(preprocessor_, best_model_):
            
            # Feature Preprocessing
            feature = prep.transform(feature)
            
            # Predict
            predict_label.append(model_.predict(feature))   
                                  
            if hasattr(model_, 'predict_proba'):
                pred_prob.append(model_.predict_proba(feature))
            elif hasattr(model_, 'decision_function'):
                pred_prob.append(model_.decision_function(feature))
            else:
                pred_prob = predict_label
        
        # Get voted predict label
        
        
        
        # Evaluation
        # acc, auc, f1, confmat, report = model.evaluate(predict_label, predict_proba, predict_label)
        # print(f"Test dataset:\nacc = {acc}\nf1score = {f1}\nauc = {auc}\n")

        return predict_proba, predict_label


if __name__ == "__main__":
    # Predict
    predict_proba, predict_label = app(data_file=data_file, metric="all", include_diagnoses=include_diagnoses, 
        num_sub=num_sub, feature_name=feature_name, label_name=label_name, model_file=model_file
    )
    
   
   
# Inputs
model_file = "./outputs.pickle"

# Load
output = pickle.load(open(model_file, "rb"))

#%% Print features after each step (search)
best_model = output["model"]
if hasattr(best_model, "best_estimator_"):
    best_model = best_model.best_estimator_

print("#"*30, "\n")
dr = best_model['dim_reduction']
print(f"Dimension reduction: feature number from {dr.n_features_} to {dr.n_components_}\n")

fs = best_model['feature_selection']
print(f"Feature selection: feature number from {fs.n_features_in_} to {fs.n_features_}\n")
print("#"*30, "\n")