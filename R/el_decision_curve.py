# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:23:07 2020

@author: lenovo
"""

import pandas as pd
import rpy2.robjects as robjects
# from rpy2.robjects import r, pandas2ri 
# from rpy2.robjects.packages import importr
 
# pandas2ri.activate()

data_file = 'D:/My_Codes/lc_private_codes/R/demo_data.xlsx'


def dca(data_file, label="label", features=["Sex", "Age"]):
    
    data = pd.read_excel(data_file)
    
    cols = [label]
    cols.extend(features)
    data_ = data[cols]
    data_.rename(columns={label: "label"}, inplace=True)
    
    indep = list(data_.columns)
    indep.remove(label)
    indep = "+".join(indep)
    format_str = label + " ~ "  + indep
    
    rscript = """
                library(readxl)
                library(ggDCA)
                library(rms)
    
                data <- read_excel('D:/My_Codes/lc_private_codes/R/demo_data1.xlsx', sheet = 1)
                data <- data.frame(data)
                
                # format_str <- parse(text = format_str)
                # f1 <- eval(format_str)
                model1 <- lrm(label ~ Sex + Age, data)
                d <- dca(model1)
                fig = ggplot(d)
                ggsave("myplot.pdf")
                ggsave(fig, file='pic_name.pdf', width=12, height=10)
        """
    
    robjects.r(rscript)
    # robjects.r.source('D:/My_Codes/lc_private_codes/R/read_excel_nomograph.R')
    # robjects.r["dca"](data_)


if __name__ ==  "__main__":
    data = dca(data_file)