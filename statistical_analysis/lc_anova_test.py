# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:33:53 2018

@author: lenovo
"""

import numpy as np

import statsmodels.api as sm

import statsmodels.formula.api as smf
from statsmodels.formula.api import ols



dat = sm.datasets.get_rdataset("Guerry", "HistData").data


moore = sm.datasets.get_rdataset("Moore", "car",cache=True)