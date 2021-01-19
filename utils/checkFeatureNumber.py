# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:13:55 2020
此代码用来打印特征的数目
@author: lichao
"""

import pickle

# Inputs
outputFile = "./outputs.pickle"

# Load
output = pickle.load(open(outputFile, "rb"))

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