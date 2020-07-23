# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 20:25:12 2018

@author: lenovo
"""

def func(msg):
    print (multiprocessing.current_process().name + '-' + msg)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4) # 创建4个进程
    for i in range(10):
        msg = "hello %d" %(i)
        pool.apply_async(func, (msg, ))
    pool.close() # 关闭进程池，表示不能在往进程池中添加进程
    pool.join() # 等待进程池中的所有进程执行完毕，必须在close()之后调用
    print ("Sub-process(es) done.")