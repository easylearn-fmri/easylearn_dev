
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:17:07 2018

@author: lenovo
"""

from collections import OrderedDict
from math import log, sqrt
import numpy as np
import pandas as pd
from six.moves import cStringIO as StringIO
from bokeh.plotting import figure, show, output_file

antibiotics = """
bacteria,                        penicillin, streptomycin, neomycin, gram
结核分枝杆菌,                      800,        5,            2,        negative
沙门氏菌,                         10,         0.8,          0.09,     negative
变形杆菌,                         3,          0.1,          0.1,      negative
肺炎克雷伯氏菌,                    850,        1.2,          1,        negative
布鲁氏菌,                         1,          2,            0.02,     negative
铜绿假单胞菌,                     850,        2,            0.4,      negative
大肠杆菌,                        100,        0.4,          0.1,      negative
产气杆菌,                         870,        1,            1.6,      negative
白色葡萄球菌,                     0.007,      0.1,          0.001,    positive
溶血性链球菌,                     0.001,      14,           10,       positive
草绿色链球菌,                     0.005,      10,           40,       positive
肺炎双球菌,                       0.005,      11,           10,       positive
"""

drug_color = OrderedDict([# 配置中间标签名称与颜色
    ("盘尼西林", "#0d3362"),
    ("链霉素", "#c64737"),
    ("新霉素", "black"),
])
gram_color = {
    "positive": "#aeaeb8",
    "negative": "#e69584",
}
# 读取数据
df = pd.read_csv(StringIO(antibiotics),
                 skiprows=1,
                 skipinitialspace=True,
                 engine='python')
width = 800
height = 800
inner_radius = 90
outer_radius = 300 - 10

minr = sqrt(log(.001 * 1E4))
maxr = sqrt(log(1000 * 1E4))
a = (outer_radius - inner_radius) / (minr - maxr)
b = inner_radius - a * maxr


def rad(mic):
    return a * np.sqrt(np.log(mic * 1E4)) + b
big_angle = 2.0 * np.pi / (len(df) + 1)
small_angle = big_angle / 7

# 整体配置
p = figure(plot_width=width, plot_height=height, title="",
           x_axis_type=None, y_axis_type=None,
           x_range=(-420, 420), y_range=(-420, 420),
           min_border=90, outline_line_color="black",
           background_fill_color="#f0e1d2")
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# annular wedges
angles = np.pi / 2 - big_angle / 2 - df.index.to_series() * big_angle  #计算角度
colors = [gram_color[gram] for gram in df.gram] # 配置颜色
p.annular_wedge(
    0, 0, inner_radius, outer_radius, -big_angle + angles, angles, color=colors,
)

# small wedges
p.annular_wedge(0, 0, inner_radius, rad(df.penicillin),
                -big_angle + angles + 5 * small_angle, -big_angle + angles + 6 * small_angle,
                color=drug_color['盘尼西林'])
p.annular_wedge(0, 0, inner_radius, rad(df.streptomycin),
                -big_angle + angles + 3 * small_angle, -big_angle + angles + 4 * small_angle,
                color=drug_color['链霉素'])
p.annular_wedge(0, 0, inner_radius, rad(df.neomycin),
                -big_angle + angles + 1 * small_angle, -big_angle + angles + 2 * small_angle,
                color=drug_color['新霉素'])

# 绘制大圆和标签
labels = np.power(10.0, np.arange(-3, 4))
radii = a * np.sqrt(np.log(labels * 1E4)) + b
p.circle(0, 0, radius=radii, fill_color=None, line_color="white")
p.text(0, radii[:-1], [str(r) for r in labels[:-1]],
       text_font_size="8pt", text_align="center", text_baseline="middle")

# 半径
p.annular_wedge(0, 0, inner_radius - 10, outer_radius + 10,
                -big_angle + angles, -big_angle + angles, color="black")

# 细菌标签
xr = radii[0] * np.cos(np.array(-big_angle / 2 + angles))
yr = radii[0] * np.sin(np.array(-big_angle / 2 + angles))
label_angle = np.array(-big_angle / 2 + angles)
label_angle[label_angle < -np.pi / 2] += np.pi  # easier to read labels on the left side

# 绘制各个细菌的名字
p.text(xr, yr, df.bacteria, angle=label_angle,
       text_font_size="9pt", text_align="center", text_baseline="middle")

# 绘制圆形，其中数字分别为 x 轴与 y 轴标签
p.circle([-40, -40], [-370, -390], color=list(gram_color.values()), radius=5)

# 绘制文字
p.text([-30, -30], [-370, -390], text=["Gram-" + gr for gr in gram_color.keys()],
       text_font_size="7pt", text_align="left", text_baseline="middle")

# 绘制矩形，中间标签部分。其中 -40，-40，-40 为三个矩形的 x 轴坐标。18，0，-18 为三个矩形的 y 轴坐标
p.rect([-40, -40, -40], [18, 0, -18], width=30, height=13,
       color=list(drug_color.values()))

# 配置中间标签文字、文字大小、文字对齐方式
p.text([-15, -15, -15], [18, 0, -18], text=list(drug_color),
       text_font_size="9pt", text_align="left", text_baseline="middle")

# show
output_file("burtin.html", title="burtin.py example")
show(p)