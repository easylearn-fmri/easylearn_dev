# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:41:05 2018

@author: lenovo
"""

from concurrent.futures import ThreadPoolExecutor
import time

# 参数times用来模拟网络请求的时间


def get_html(times):
    time.sleep(times)
    print("get page {}s finished\n".format(times))
    return times


with ThreadPoolExecutor(2) as executor:
    #executor = ThreadPoolExecutor(max_workers=2)
    # 通过submit函数提交执行的函数到线程池中，submit函数立即返回，不阻塞
    task1 = executor.submit(get_html, (0.5))
    task2 = executor.submit(get_html, (0.5))

