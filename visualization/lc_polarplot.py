# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:39:53 2018
polar bar
@author: lenovo
"""

from pyecharts import Polar
radius1 =['Left orbital frontal cortex', 
         'Right orbital frontal cortex', 
         'Left dorsolateral prefrontal cortex',
         'Left angular gyrus']

radius2 =['Right primary somatosensory cortex',	
'Left primary somatosensory cortex'	,
'Right supplementary motor area',
'Left primary auditory cortex',
'Left thalamus',
'Right visual association cortex',
'Right primary association visual cortex',
'Left visual association cortex'
]

polar =Polar("极坐标系-堆叠柱状图示例", width=1200, height=600)
polar.add("HC", [0.9803,0.9691, 0.9579,0.8857, 0.7264,0.9935,1.1036,0.9253], 
              radius_data=radius2, type='barAngle', is_stack=False)

polar.add("MDD", [0.9589, 0.9398,    0.9183,    0.8551,    0.7138,    0.9678,    1.0596,    0.9105],
              radius_data=radius1, type='barAngle', is_stack=False)

polar.add("BD", [0.9414,    0.9218,    0.9307,    0.8490,    0.6764 ,   0.9652 ,   1.0598 ,   0.8775], 
              radius_data=radius1, type='barAngle', is_stack=False)


polar.add("SZ", [ 0.9028,    0.8883,    0.8804  ,  0.8272  ,  0.6668 ,  0.9072  ,  1.0102,    0.8547],
              radius_data=radius1, type='barAngle', is_stack=False)

polar.show_config()
polar.render()