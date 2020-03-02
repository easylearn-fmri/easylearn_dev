# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:39:48 2019
This script is used to perform binomial test for classification performances, e.g., accuracy, sensitivity, specificity.
@author: lenovo
"""
from scipy.special import comb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['savefig.dpi'] = 600 #图片像素
# plt.rcParams['figure.dpi'] = 600 #分辨率

n =  140
k = 119
p1 = 0.5
p2 = 0.5

def lc_binomialtest(n, k, p1, p2):
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

def lc_plot(prob, k, p, titlename):
    plt.plot(prob)
    if p < 0.001:
        plt.title(titlename + f'\np < 0.001',fontsize=10)
    else:
        plt.title(titlename + '\n' + 'p = ' + '%.3f' %p, fontsize=10)

    plt.plot([k,k],[prob[k],prob[k]],'.', markersize=10)
    plt.plot([k,k],[0,0.06],'--', markersize=15)
    # plt.title(titlename,fontsize=10)
    plt.xlabel('Number of correct predictions',fontsize=8)
    plt.ylabel('Probability', fontsize=8)
    plt.show()


if __name__ ==  "__main__":
    k = 38
    p, sum_prob, prob, randk = lc_binomialtest(63, 39, 0.5, 0.5)
    print(p)
    lc_plot(prob, k, p, titlename = f'Drug naive\n (totle = 63\n True positive = 39)')


    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
    #                     wspace=0.5, hspace=0.5)
    
    # p1, p2 = 0.5, 0.5

    # n, k = 149, 119 
    # p, sum_prob, prob, randk = lc_binomialtest(140, 119, p1, p2)
    # plt.subplot(2, 5, 1)
    # lc_plot(prob, k, p, titlename = f'Training set (n = 149)')

    # n, k = 61, 49
    # p, sum_prob, prob, randk = lc_binomialtest(n, k, p1, p2)
    # plt.subplot(2, 5, 2)
    # lc_plot(prob, k, p, titlename = f'Test set (n = 61)')

    # n, k = 6, 4
    # p, sum_prob, prob, randk = lc_binomialtest(n, k, p1, p2)
    # plt.subplot(2, 5, 3)
    # lc_plot(prob, k, p, titlename = f'GE (n = 6)')

    # n, k = 24, 18
    # p, sum_prob, prob, randk = lc_binomialtest(n, k, p1, p2)
    # plt.subplot(2, 5, 4)
    # lc_plot(prob, k, p, titlename = f'Philips (n = 24)')

    # n, k = 11, 10
    # p, sum_prob, prob, randk = lc_binomialtest(n, k, p1, p2)
    # plt.subplot(2, 5, 5)
    # lc_plot(prob, k, p, titlename = f'Siemens (n = 11)')

    # n, k = 20,16 
    # p, sum_prob, prob, randk = lc_binomialtest(n, k, p1, p2)
    # plt.subplot(2, 5, 6)
    # lc_plot(prob, k, p, titlename = f'Toshiba (n = 20)')

    # n, k = 4, 3
    # p, sum_prob, prob, randk = lc_binomialtest(n, k, p1, p2)
    # plt.subplot(2, 5, 7)
    # lc_plot(prob, k, p, titlename = f'1 mm (n = 4)')

    # n, k = 3, 2
    # p, sum_prob, prob, randk = lc_binomialtest(n, k, p1, p2)
    # plt.subplot(2, 5, 8)
    # lc_plot(prob, k, p, titlename = f'2 mm (n = 3)')

    # n, k = 21, 15
    # p, sum_prob, prob, randk = lc_binomialtest(n, k, p1, p2)
    # plt.subplot(2, 5, 9)
    # lc_plot(prob, k, p, titlename = f'5 mm (n = 21)')

    # n, k = 33, 28
    # p, sum_prob, prob, randk = lc_binomialtest(n, k, p1, p2)
    # plt.subplot(2, 5, 10)
    # lc_plot(prob, k, p, titlename = f'8 mm (n = 33)')
    
    # # plt.savefig(r'D:\stats_all.tif', dpi=600)
    # plt.show()
