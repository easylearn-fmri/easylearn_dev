# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:13:55 2020
此代码用来加载机器学习的结果
@author: lichao
"""

import pickle

# Inputs
output1 = "./outputs.pickle"
output2 = "./stat.pickle"

# Load
model = pickle.load(open(output1, "rb"))
stat = pickle.load(open(output2, "rb"))

# Print keys
print(model.keys())
print(stat.keys())

# Get outputs
pvalue_acc = stat["pvalue_acc"]
print(f"pvalues of accuracy is {pvalue_acc}")