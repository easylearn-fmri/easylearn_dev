# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:44:22 2018

@author: lenovo
"""

def funcA(A):
    print("function A")

def funcB(B):
#    print(B(2))
    print("function B")

@funcB
def func(c):
    print("function C")
    return c**2