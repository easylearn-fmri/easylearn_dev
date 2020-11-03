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


def dca(data_file, label="label", features=["FrequencySize", "MaxIntensity"]):
    
    data = pd.read_excel(data_file)
    
    cols = [label]
    cols.extend(features)
    data_ = data[cols]
    data_.rename(columns={label: "label"}, inplace=True)
    
    # robjects.r(
    #     r"""dca <- function(data, legend="baseline model"){
    #         library(readxl)
    #         library(rms)
    #         library(survival)

    #         baseline.model <- decision_curve(label ~ FrequencySize +  MaxIntensity,
    #                                          data = data, 
    #                                          bootstraps = 50,
    #                                          confidence.intervals=NA)
            
    #         #plot the curve
    #         plot_decision_curve(baseline.model,  curve.names = legend)
    #         }"""
    # )
    
    rfun = """
        nomo <- function(data){
            data = as.data.frame(data)
            
            ## 第三步 按照nomogram要求“打包”数据，绘制nomogram的关键步骤,??datadist查看详细说明
            dd=datadist(data)
            options(datadist="dd") 
            
            f1 = label ~ FrequencySize + MaxIntensity
            f <- lrm(f1, data = data)
            
            nom <- nomogram(f, fun=plogis, lp=F, funlabel="Risk")
            plot(nom)

        }
    """
    
    robjects.r['nomo'](data_)
    # robjects.r(rscript)
    # robjects.r.source('D:/My_Codes/lc_private_codes/R/read_excel_nomograph.R')
    # robjects.r["dca"](data_)


if __name__ ==  "__main__":
    data = dca(data_file)