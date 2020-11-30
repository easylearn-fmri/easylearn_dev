# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:13:55 2020

@author: lenovo
"""

import pickle
import numpy as np

rs = pickle.load(open("./outputs.pickle", "rb"))
print(rs.keys())

