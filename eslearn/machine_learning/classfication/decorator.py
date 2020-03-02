# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:45:12 2019

@author: lenovo
"""


def my_reshape(func):
    def wrapper(*args, **kwargs):
        args=[ar[1] for ar in args]
        return func(*args, **kwargs)
    return wrapper


@my_reshape
def say_hello(a,b):
    print(a+b)


if __name__ == "__main__":
    say_hello('aa','bb')
