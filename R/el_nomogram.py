# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:23:07 2020

@author: lenovo
"""

import pandas as pd
import rpy2.robjects as robjects


def plot_nomogram(data_file, label="label", features=["FrequencySize", "MaxIntensity"]):
    """Plot nomogram using R language

    Parameters:
    ----------
    data_file: file string
        File that contains source data

    label: string
        Column name of dependent variable

    features: string list
        Column names of independent variables/predictors
    """

    data = pd.read_excel(data_file)
    cols = [label]
    cols.extend(features)
    data_ = data[cols]
    data_.rename(columns={label: "label"}, inplace=True)
    
    rfun = """
        nomo <- function(data){
            data = as.data.frame(data)
            dd=datadist(data)
            options(datadist="dd") 
            
            f1 = label ~ .
            f <- glm(f1, family = binomial(), data = data)
            
            nom <- nomogram(f, fun=plogis, lp=F, funlabel="Risk")
            plot(nom)

        }
    """
    
    robjects.r['nomo'](data_)


if __name__ ==  "__main__":
    data_file = 'D:/My_Codes/lc_private_codes/R/demo_data.xlsx'
    data = plot_nomogram(data_file)