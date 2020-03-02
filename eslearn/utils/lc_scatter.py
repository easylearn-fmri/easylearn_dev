# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:25:00 2018
画散点图和拟合直线
@author: LiChao
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_scatter_plus_fittingLine(x, y, title='polyfitting'):
    # 绘制散点
    # 直线拟合与绘制
    x = np.reshape(x, [len(x)])
    y = np.reshape(y, [len(y)])
    z1 = np.polyfit(x, y, 1)  # 用1次多项式拟合
    yvals = np.polyval(z1, x)
#    p1 = np.poly1d(z1)
#    yvals=p1(x)#也可以使用
    plt.plot(x, y, 'o', markersize=6, label='original values')
    plt.plot(x, yvals, 'r', linestyle=':', label='polyfit values')
#    plt.xlabel('x axis')
#    plt.ylabel('y axis')
    plt.legend(loc=0)  # 指定legend的位置
    plt.title(title)
    plt.show()
#    plt.savefig('p1.png')
