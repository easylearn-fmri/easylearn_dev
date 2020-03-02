# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:59:28 2018

@author: lenovo
"""

from pyecharts import Polar,Radar


#radius =['周一', '周二', '周三', '周四', '周五', '周六', '周日']
#polar =Polar("极坐标系-堆叠柱状图示例", width=1200, height=600)
#polar.add("", [1, 2, 3, 4, 3, 5, 1], radius_data=radius, type='barAngle', is_stack=True)
#polar.add("", [2, 4, 6, 1, 2, 3, 1], radius_data=radius, type='barAngle', is_stack=True)
#polar.add("", [1, 2, 3, 4, 1, 2, 5], radius_data=radius, type='barAngle', is_stack=True)
#polar.show_config()
#polar.render()


value_bj =[ [55, 9, 56, 0.46, 18, 6, 1], [25, 11, 21, 0.65, 34, 9, 2], [56, 7, 63, 0.3, 14, 5, 3], [33, 7, 29, 0.33, 16, 6, 4]]
value_sh =[ [91, 45, 125, 0.82, 34, 23, 1], [65, 27, 78, 0.86, 45, 29, 2], [83, 60, 84, 1.09, 73, 27, 3], [109, 81, 121, 1.28, 68, 51, 4]]
c_schema=[{"name": "AQI", "max": 300, "min": 5}, {"name": "PM2.5", "max": 250, "min": 20}, {"name": "PM10", "max": 300, "min": 5}, {"name": "CO", "max": 5}, {"name": "NO2", "max": 200}, {"name": "SO2", "max": 100}]
radar =Radar()
radar.config(c_schema=c_schema, shape='circle')
radar.add("北京", value_bj, item_color="#f9713c", symbol=None)
radar.add("上海", value_sh, item_color="#b3e4a1", symbol=None)
radar.show_config()
radar.render()


from pyecharts import Radar
schema =[ ("销售", 6500), ("管理", 16000), ("信息技术", 30000), ("客服", 38000), ("研发", 52000), ("市场", 25000)]
v1 =[[4300, 10000, 28000, 35000, 50000, 19000]]
v2 =[[5000, 14000, 28000, 31000, 42000, 21000]]
radar =Radar()
radar.config(schema)
radar.add("预算分配", v1, label_color=["#4e79a6"],is_splitline=True, is_axisline_show=True)
radar.add("实际开销", v2, label_color=["r"], is_area_show=True)
radar.show_config()
radar.render()