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
from eslearn.model_evaluator import ModelEvaluator


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
        output = pickle.load(open(self.model_file, "rb"))
        preprocessor = output["preprocessor"]
        best_model = output["model"]
        if hasattr(best_model, "best_estimator_"):
            best_model = best_model.best_estimator_

        # Get data
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
        final_label = self.vote(predict_label)
        final_prob = np.mean(pred_prob,0)
        
        # Evaluation
        acc, sens, spec, _, _ = ModelEvaluator().binary_evaluator(
            target, final_label, final_prob,
            verbose=1, is_showfig=False, is_savefig=False
        )

        return acc, sens, spec

    @staticmethod
    def vote(labels):
        labels = np.array(labels)
        [n_models, n_cases] = np.shape(labels)
        final_label = []
        for i in range(n_cases):
            num = labels[:,i]
            result = {}
            for j in range(n_models):
                if result.get(labels[j,i]) == None:
                    result[labels[j,i]] = 1
                else:
                    result[labels[j,i]] += 1
            
            ct = -np.inf
            for k in result.keys():
                if result[k] > ct:
                    ct = result[k]
                    fl = k
            final_label.append(fl)
        
        return np.array(final_label)
                    


if __name__ == "__main__":   
  model_file = r"F:\线上讲座\demo_data\szVShc_fc\outputs.pickle"
  data_file = r'D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\machine_learning\tests/dataLoadingTest.json'
     
  pred = Predict(data_file, model_file)
  acc, sens, spec = pred.run()