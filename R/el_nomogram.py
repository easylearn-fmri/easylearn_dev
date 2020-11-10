# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:23:07 2020

@author: Li Chao
"""

import os
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()


def plot_nomogram(data_file, label="label", features=["Sex", "Age"]):
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
    
    indep = list(data_.columns)
    indep.remove(label)
    indep = "+".join(indep)
    format_str = label + " ~ "  + indep
    
    rfolder = os.path.dirname(__file__)
    # print(rfolder)

    
    rscript = """
        nomo <- function(data, format_str, rfolder){
            
            dd=datadist(data)
            options(datadist="dd")
            # setwd("D:/My_Codes/virtualenv_eslearn/Lib/site-packages/eslearn/R")
            setwd(rfolder)
            source("nomograph.R")
            nomo(data, format_str)
        }
        """
    
    rs = r"""
        library(readxl)
        source("datadist")
        # setwd("D:/My_Codes/virtualenv_eslearn/Lib/site-packages/eslearn/R")
        # source("nomograph.R")
        
        data <- read_excel('D:/My_Codes/lc_private_codes/R/demo_data1.xlsx', sheet = 1)
        dd=datadist(data)
        options(datadist="dd")
        # format_str = "label ~ Sex + Age"
        # nomo(data, format_str)
    """
    
    robjects.globalenv['data_'] = data_
    # robjects.r["nomo"](data_, format_str)
    robjects.r(rs)
    
    # os.chdir(cwd)


if __name__ ==  "__main__":
    data_file = 'D:/My_Codes/lc_private_codes/R/demo_data.xlsx'
    data = plot_nomogram(data_file)
    

    