# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:39:48 2019
This script is used to perform binomial test for classification performances, e.g., accuracy, sensitivity, specificity.
@author: lenovo
"""
from scipy.special import comb
import matplotlib.pyplot as plt
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np


def binomialtest(n, k, p):
    p = 1-binom.cdf(k, n, p) 
    return p

def lc_plot(n, k, p, titlename, outname=None):
    # plt.plot(prob)
    # if p < 0.001:
    #     plt.title(titlename + f'\np < 0.001',fontsize=10)
    # else:
    #     plt.title(titlename + '\n' + 'p = ' + '%.3f' %p, fontsize=10)

    # plt.plot([k,k],[prob[k],prob[k]],'.', markersize=10)
    # plt.plot([k,k],[0,0.06],'--', markersize=15)
    # plt.xlabel('Number of correct predictions',fontsize=8)
    # plt.ylabel('Probability', fontsize=8)
    
    # fig, ax = plt.subplots(1, 1, figsize = (a, b))
    
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    x = np.arange(binom.ppf(0.01, n, p),
                  binom.ppf(0.99, n, p))
    ax.plot(x, binom.pmf(x, n, p), 'bo', color='gray', ms=5, label='binomial probability mass function')
    ax.vlines(x, 0, binom.pmf(x, n, p), colors='gray', lw=5, alpha=0.5)
    
    rv = binom(n, p)
    ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
            label='frozen probability mass function')


    # plt.plot([k,k],[prob[k],prob[k]],'.', markersize=10)
    # plt.plot([k,k],[0,0.06],'--', markersize=15)

    # ax.legend(loc='lower right', frameon=False)
    num1 = 1.1
    num2 = 1
    num3=1
    num4=0.5
    ax.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    plt.title(titlename)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    if outname: plt.savefig(outname)

    plt.show()



if __name__ ==  "__main__":
    n = 1072
    acc = 0.97
    k = np.int32(n * acc)
    print(k)
    p = binomialtest(n, k, 0.5)
    import pandas as pd
    dd = pd.DataFrame(np.random.randn(10,10))
    print(dd)
    lc_plot( n, k, 0.5, titlename = f'Testing data\n (Sample size = {n}, Accuracy = {acc})')
    plt.savefig("./binomialDistribution.pdf")
