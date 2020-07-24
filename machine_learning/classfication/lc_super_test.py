# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:43:09 2019

@author: lenovo
"""

class Base(object):
       def __init__(self):
              print ("Base init")
 
class Medium1(Base):
       def __init__(self):
              Base.__init__(self)
              print ("Medium1 init")
 
class Medium2(Base):
       def __init__(self):
              Base.__init__(self)
              print ("Medium2 init")
 
class Leaf(Medium1, Medium2):
       def __init__(self):
              Medium1.__init__(self)
              Medium2.__init__(self)
              print ("Leaf init")    
              
leaf = Leaf()