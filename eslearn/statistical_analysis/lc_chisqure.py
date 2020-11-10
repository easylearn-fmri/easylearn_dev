# -*- coding: utf-8 -*-
"""Chi-square test for independence of variables in a contingency table or constituent ratio.

NOTE. When 'correction' is False in lc_chi2, then lc_chi2 and lc_chisqure is the same.
@author: Li Chao
Email:lichao19870617@gmail.com OR lichao19870617@163.com
"""
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import numpy as np


def lc_chi2(obs, tt, **kwargs):
    """
    Parameters:
    -----------
        obs: list with each item is a number
            observed frequence
        tt: list with each item is a number
                total number of each group
    Returns:
    --------
        chisqurevalue: float
            chi-square value
        p: float
            p-value
    EXAMPLE:
    tt = [120, 81]
    obs = [31, 67]
    chi2value, pvalue = lc_chi2(
    print(f"chi-squre value = {chi2value}\np-value = {pvalue}")
    """

    data = np.array([obs, np.array(tt) - np.array(obs)])
    results = chi2_contingency(data)
    chi2value = results[0]
    pvalue = results[1]
    return (chi2value, pvalue)


def lc_chisqure(obs, tt):
    """Chi-square test for constituent ratio

    Parameters:
    -----------
        obs: list with each item is a number
            observed frequence
        tt: list with each item is a number
                total number of each group
    Returns:
    --------
        chisqurevalue: float
            chi-square value
        p: float
            p-value
    EXAMPLE:
    tt = [120, 81]
    obs = [31, 67]
    chisqurevalue, p = lc_chisqure(obs, tt)
    print(f"chi-squre value = {chisqurevalue}\np-value = {p}")
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
    tt = [120, 81]
    obs = [31, 67]

    chi2value, pvalue = lc_chi2(obs, tt,  correction=False)
    print(f"chi-squre value = {chi2value}\np-value = {pvalue}")

    chisqurevalue, p = lc_chisqure(obs, tt)
    print(f"chi-squre value = {chisqurevalue}\np-value = {p}")
