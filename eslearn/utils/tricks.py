# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:12:52 2018

@author: lenovo
"""
#import numpy as np
seq = [1, 2, 3, 5]
# 类似于matlab的@，定义一个函数


def f(x): return pow(x, 3)


f(2)
# lambda配合map来对一个列表遍历求值
myMap = map(f, seq)
print(list(myMap))
# lambda 配合filter函数
result = list(filter(lambda x: x >= 2, seq))
print(result)
