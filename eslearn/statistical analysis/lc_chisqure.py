# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:31:57 2018
构成比的卡方检验
Chi-square test of independence of variables in a contingency table.
@author: lenovo
"""
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import numpy as np


def lc_chi2(data):
    # data=np.array([[10,20],[10,30],[20,400]])
    results = chi2_contingency(data)
    chi2value = results[0]
    pvalue = results[1]
    return (chi2value, pvalue)


def lc_chisqure(obs, tt):
    """
    obs: observed frequence
    tt: total number of each group
    NOTE. Make sure the number are np.array
    The results is in line with SPSS
    """
    tt = np.array(tt)
    obs1 = obs
    obs1 = np.array(obs1)
    obs2 = tt - obs1
    obs_all = np.vstack([obs1, obs2]).T
    n_row = np.shape(obs_all)[0]
    n_col = np.shape(obs_all)[1]
    df = (n_row - 1) * (n_col) / 2  # free degree

    frq1 = np.sum(obs1) / np.sum(tt)
    frq2 = np.sum(obs2) / np.sum(tt)
    f_exp1 = tt * frq1
    f_exp2 = tt * frq2
    f_exp = np.vstack([f_exp1, f_exp2]).T

    chisqurevalue = np.sum(((obs_all - f_exp)**2) / f_exp)
    p = (1 - chi2.cdf(chisqurevalue, df=df))
    return chisqurevalue, p


if __name__ == "__main__":
    tt = [120, 81 - 20]
    obs = [31, 67 - 20]
    print(lc_chisqure(obs, tt))
