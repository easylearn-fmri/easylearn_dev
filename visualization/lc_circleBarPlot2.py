# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:29:21 2018

@author: lenovo
"""

from collections import OrderedDict
from math import log, sqrt
import numpy as np
import pandas as pd
from six.moves import cStringIO as StringIO
from bokeh.plotting import figure, show, output_file


#df=np.abs(pd.DataFrame(np.random.randn(5,3),index=['age','sex','height','weigh','BMI'], columns=['SZ','BD','MDD'])*50)
df =  pd.read_excel(r'D:\others\彦鸽姐\0630.xlsx')
df = df.set_index(df.columns[0])
df = df.iloc[:,[1,3,5]]
sca = 100
df = df * sca

# 统计数据维度
nCol=np.shape(df)[1]
nRow=np.shape(df)[0]
nameCol=df.columns
maxNum=np.max(df.values)
minNum=np.min(df.values)
nCircle=5  # 画几条等高线

# 图像基本配置
width = 2000
height = 2000
inner_radius = 100
outer_radius = 200

#配置各个扇形区域的颜色
#fanShapedColors = pd.DataFrame([['bisque']*3+['lightgreen']*4+['goldenrod']*2+
#                   ['skyblue']*6+['thistle']*5+['pink']*1]) # 配置颜色
#fanShapedColors =[fanShapedColors.iloc[0,x] for x in range(len(df))] 

fanShapedColors = ['w']*len(df)

# 配置每个bar的颜色
barColors = ['green','red','peru'] # 配置颜色
# =============================================================================

big_angle = 2.0 * np.pi / (len(df) + 1)
small_angle = big_angle / nCircle

# 整体配置
p = figure(plot_width=width, plot_height=height, title="",
           x_axis_type=None, y_axis_type=None,
           x_range=(-420, 420), y_range=(-420, 420),
           min_border=0, outline_line_color="black",
           background_fill_color="#f0e1d2")
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# annular wedges
angles = np.pi / 2 - big_angle / 2 - np.arange(0,nRow,1) * big_angle  #计算角度

p.annular_wedge(0, 0, inner_radius, outer_radius,
                -big_angle + angles, angles, color=fanShapedColors)
#
# small wedges

p.annular_wedge(0, 0, inner_radius, inner_radius+df.iloc[:,0],
                -big_angle + angles + 1 * small_angle, -big_angle + angles + 1.5 * small_angle,
                color=barColors[0])

p.annular_wedge(0, 0, inner_radius, inner_radius+df.iloc[:,1],
                -big_angle + angles +  2* small_angle, -big_angle + angles + 2.5 * small_angle,
                color=barColors[1])

p.annular_wedge(0, 0, inner_radius, inner_radius+df.iloc[:,2],
                -big_angle + angles + 3 * small_angle, -big_angle + angles + 3.5 * small_angle,
                color=barColors[2])
#
#p.annular_wedge(0, 0, inner_radius, inner_radius+df.iloc[:,3],
#                -big_angle + angles + 4 * small_angle, -big_angle + angles + 4.5 * small_angle,
#                color=barColors[3])
# =============================================================================
## 绘制等高线和刻度
radii = np.around(np.linspace(inner_radius,outer_radius,nCircle), decimals=1)
labels = radii / sca
p.circle(0, 0, radius=radii, fill_color=None, line_color="white")
p.text(0, radii[:-1], [str(r) for r in labels[:-1]],
       text_font_size="10pt", text_align="center", text_baseline="middle")

## 分割线
p.annular_wedge(0, 0, inner_radius, outer_radius + 10,
                -big_angle + angles, -big_angle + angles, color="black")
### 细菌标签
xr = (50+radii[len(radii)-1]) * np.cos(np.array(-big_angle / 2 + angles))
yr = (50+radii[len(radii)-1]) * np.sin(np.array(-big_angle / 2 + angles))
label_angle = np.array(-big_angle / 2 + angles)
label_angle[label_angle < -np.pi / 2] += np.pi  # easier to read labels on the left side

## 绘制各个细菌的名字
p.text(xr, yr, df.index, angle=label_angle,
       text_font_size="12pt", text_align="center", text_baseline="middle")

#扇形区域颜色的legend
p.circle([-40, -40], [-370, -390], color=np.unique(list(fanShapedColors)), radius=5)

p.text([-30, -30], [-370, -390], text=["Gram-" + gr for gr in barColors],
       text_font_size="7pt", text_align="left", text_baseline="middle")

# 分组的legend 
p.rect([-30, -30, -30,-30], [20, 10, 0,-10], width=20, height=8,
       color=list(barColors))
# 配置中间标签文字、文字大小、文字对齐方式
p.text([-15, -15, -15,-15], [20, 10, 0,-10], text=list(nameCol),
       text_font_size="10pt", text_align="left", text_baseline="middle")

# ============================================================================
# show
output_file("circleBar.html", title="circleBar")
show(p)