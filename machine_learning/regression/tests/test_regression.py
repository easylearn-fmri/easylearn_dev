#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Author: Mengshi Dong <dongmengshi1990@163.com>
"""

import time
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, max_error

from eslearn.base import BaseMachineLearning, DataLoader
from eslearn.machine_learning.regression._base_regression import BaseRegression
from eslearn.machine_learning.regression.regression import Regression
from eslearn.model_evaluator import ModelEvaluator


x, y = datasets.make_regression(n_samples=500, n_informative=50, n_features=500, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

x1 = pd.DataFrame([])
x1["__ID__"] = [i for i in range(500)]
ff = pd.concat([x1, pd.DataFrame(x)], axis=1)
ff.to_csv(r"F:\一月份线上讲座\features.csv", index=False)

y1 = pd.DataFrame([])
y1["__ID__"] = [i for i in range(500)]
y1["__Targets__"] = y
y1.to_csv(r"F:\一月份线上讲座\targets.csv", index=False)

def test_regression():
    time_start = time.time()
    clf = Regression(configuration_file=r'D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\GUI\tests\configuration_file_reg.json',
                     out_dir=r"D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\GUI\tests")
    clf.main_run()
    clf.permutation_test()
    time_end = time.time()
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)


if __name__ == "__main__":
    test_regression()