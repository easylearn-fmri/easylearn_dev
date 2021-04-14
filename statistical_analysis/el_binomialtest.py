# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:39:48 2019
This script is used to perform binomial test for classification performances, 
e.g., accuracy, sensitivity, specificity.
@author: Li Chao
Email: lichao19870617@163.com
"""
from scipy.special import comb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def binomialtest(n, k, p1, p2):
    sum_prob = 0
    prob = []
    randk = 0
    for i in range(k):
    	sum_prob += comb(n,i) * pow(p1,i) * pow(p2, (n-i))
    	if (sum_prob >= 0.95) and (not randk):
    		randk = i+1
    		print(f'sum_prob in {randk} is {sum_prob}')

    p = 1 - sum_prob if sum_prob <=1 else 0

    for i in range(n):
    	prob.append(comb(n,i) * pow(p1,i) * pow(p2,(n-i)))

    return p, sum_prob, prob, randk

def lc_plot(prob, k, p, titlename, outname=None):
    plt.figure(figsize=(4,3))
    plt.plot(prob, color='darkcyan')
    if p < 0.001:
        plt.title(titlename + f'\np < 0.001',fontsize=12)
    else:
        plt.title(titlename + '\n' + 'p = ' + '%.3f' %p, fontsize=12)

    plt.plot([k,k],[prob[k],prob[k]],'.', markersize=10, color='darkcyan')
    plt.plot([k,k],[0,0.06],'--', markersize=15, color='darkturquoise')
    plt.xlabel('Number of correct predictions', fontsize=10)
    plt.ylabel('Probability', fontsize=10)

    ax = plt.gca()
    ax.spines['bottom'].set_position(('axes', 0))
    ax.spines['left'].set_position(('axes', 0))
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
    if outname:
        plt.savefig(outname)
    plt.show()


if __name__ ==  "__main__":
    n = 38
    acc = 0.79
    k = np.int32(n * acc)
    print(k)
    p, sum_prob, prob, randk = binomialtest(n, k, 0.5, 0.5)
    print(p)
    lc_plot(prob, k, p, 
        titlename = f'Binomial test\n (Sample size = {n}, Accuracy = {acc})',
        outname=r"F:\yueyingkeji\PD_tiantan\codes\resultls_classification\BinomialTest_all.pdf")
