# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:55:08 2018

@author: lenovo
"""

from pyecharts import Pie
import numpy as np
attr = ['male','female']
v1 = sex4Num
pie = Pie('饼图示例')
pie.add('',attr,v1,is_label_show = True)
pie.render(r'D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\pie01.html')

#
df = pd.DataFrame({'HC': sex1Num,\
                   'MDD': sex2Num,\
                    'SZ':sex3Num,\
                    'BD': sex4Num},\
                   index=['male', 'female'])
plot = df.plot.pie(y='BD', figsize=(5, 5))  