# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:13:55 2020

@author: lenovo
"""

import pickle
import numpy as np

md = pickle.load(open("./outputs.pickle", "rb"))
stat = pickle.load(open("./stat.pickle", "rb"))

print(md.keys())
print(stat.keys())


