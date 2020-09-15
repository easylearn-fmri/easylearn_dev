# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:57:46 2020

@author: lenovo
"""

from timer import timer
import time


@timer
def function():
    time.sleep(1)
        
if __name__ == "__main__":
    function()