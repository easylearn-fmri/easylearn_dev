# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:21:28 2019

@author: lenovo
"""

import matplotlib.pyplot as plt
import networkx as nx
G=nx.path_graph(400)
G.add_path([10,11,12])
nx.draw(G,with_labels=True,label_size=1000,node_size=1000,font_size=20)
plt.show()

for c in sorted(nx.connected_components(G),key=len,reverse=True):
    print(c)      #看看返回来的是什么？结果是{0,1,2,3}
    print(type(c))   #类型是set
    print(len(c))   #长度分别是4和3（因为reverse=True，降序排列）
 
 
largest_components=max(nx.connected_components(G),key=len)  # 高效找出最大的联通成分，其实就是sorted里面的No.1
print(largest_components)  #找出最大联通成分，返回是一个set{0,1,2,3}
print(len(largest_components))  #4