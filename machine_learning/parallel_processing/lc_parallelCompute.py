# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 19:47:54 2018

@author: lenovo
"""
#from joblib import Parallel, delayed
import time
import numpy as np
from math import sqrt
# small data
%time result1 = Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10000))
%time result2 = Parallel(n_jobs=8)(delayed(sqrt)(i**2) for i in range(10000))

#big data
%time result = Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(1000000))
%time result = Parallel(n_jobs=-1)(delayed(sqrt)(i**2) for i in range(1000000))


def add(x):
    np.zeros([x,1])

start_time=time.clock()
a=Parallel(n_jobs=2,backend="threading")(delayed(add)(i) for i in range(100000))
end_time=time.clock()
print('耗时{:.1f}秒'.format(end_time-start_time))  
