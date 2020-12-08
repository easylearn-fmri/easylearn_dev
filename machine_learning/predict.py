# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 21:55:50 2020
This script is designed to predict unseen data using pre-trained model.
@author: Li Chao, Dong Mengshi
"""

import numpy as np
import pandas as pd
import pickle

from preprocess import preprocess
from preprocess import METRICS
from ensemble_model import Model


def app(data_file=None, 
        metric="all",
        include_diagnoses=(1,3), 
        num_sub=None, 
        feature_name=None, 
        label_name=None, 
        model_file=None):

    """ Application of our model

    Parameters:
    ----------
    data_file: file path
        Where is the .mat data file
    
    metric: string
        metric name, such as "FA" or "MD" or "all" [all modalities]

    include_diagnoses: tuple or list
        which groups/diagnoses included in training and validation, such as (1,3)

    num_sub: int
        How many subjects

    feature_name: string
        Name of the feature, such as "test_set"
    
    label_name: string
        Name of the label, such as "test_diagose"

    model_file: file path
        Where is the model file
    
    Returns:
    -------
    predict_proba: ndarray with shape of [n_samples, [*n_classes]]
        predict probability

    predict_label: ndarray with shape of [n_samples, ]
        predict label
    """
    
     
    # Load model
    all_models = pickle.load(open(model_file, "rb"))
    
    # Get data and preprocessing
    
    # Feature Preprocessing
    
    # Predict
    predict_proba, predict_label = model.predict(all_models["merged_model"], data)

    # Evaluation
    # acc, auc, f1, confmat, report = model.evaluate(predict_label, predict_proba, predict_label)
    # print(f"Test dataset:\nacc = {acc}\nf1score = {f1}\nauc = {auc}\n")

    return predict_proba, predict_label


if __name__ == "__main__":
    # Predict
    predict_proba, predict_label = app(data_file=data_file, metric="all", include_diagnoses=include_diagnoses, 
        num_sub=num_sub, feature_name=feature_name, label_name=label_name, model_file=model_file
    )
    
   