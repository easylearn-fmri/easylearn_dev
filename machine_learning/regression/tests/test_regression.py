#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Author: Mengshi Dong <dongmengshi1990@163.com>
"""

import time
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from eslearn.base import BaseMachineLearning, DataLoader
from eslearn.machine_learning.regression.regression import Regression


# x, y = datasets.make_regression(n_samples=500, n_informative=3, n_features=5, random_state=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# x1 = pd.DataFrame([])
# x1["__ID__"] = [i for i in range(500)]
# ff = pd.concat([x1, pd.DataFrame(x)], axis=1)
# ff.to_csv("./features.csv", index=False)

# y1 = pd.DataFrame([])
# y1["__ID__"] = [i for i in range(500)]
# y1["__Targets__"] = y
# y1.to_csv("./targets.csv", index=False)

def test_regression():
    time_start = time.time()
    reg = Regression(configuration_file="./regression_configuration.json",
                     out_dir="./")
    reg.main_run()
    print(reg.method_model_evaluation_)
    reg.run_statistical_analysis()

    time_end = time.time()
    assert isinstance(reg.outputs, dict)
    print(f"Running time = {time_end-time_start}\n")
    print("="*50)


if __name__ == "__main__":
    test_regression()