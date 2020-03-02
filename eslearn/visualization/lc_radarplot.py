# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:37:41 2018

@author: lenovo
"""

import pygal

radar_chart = pygal.Radar()
pygal.Radar(fill=True,line='-') 
radar_chart.title = 'radar_plot'
radar_chart.x_labels = ['Richards', 'DeltaBlue', 'Crypto', 'RayTrace', 'EarleyBoyer', 'RegExp', 'Splay', 'NavierStokes']
radar_chart.add('Chrome', [6395, 8212, 7520, 7218, 12464, 1660, 2123, 8607])
radar_chart.add('Firefox', [7473, 8099, 11700, 2651, 6361, 1044, 3797, 9450])
radar_chart.add('Opera', [3472, 2933, 4203, 5229, 5810, 1828, 9013, 4669])
radar_chart.add('IE', [43, 41, 59, 79, 144, 136, 34, 102])
radar_chart.render_to_file('bar_chart.svg')

#bar = pygal.Bar()
#bar.title = "bar测试"
#bar.x_labels = ["1", "2"]
#bar.add("webp", [20, 30])
#bar.add("jpg", [20, 30])
#bar.render_to_file()
#bar.render_to_png(r'D:\myCodes\MVPA_LIChao\MVPA_Python\plot\a.png')