# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:14:22 2018
财务
@author: lenovo
"""
# import
import pandas as pd
import numpy as np
# input
textFile = r'I:\其他文件\老舅财务\李锰等\201702\201702-1.txt'
targetName = '张华'
targetItem = ['户名', '帐号', '交易日期', '摘要',
              '借贷标志', '交易金额', '投资人', '卡号']


def extractDataFromTxt():
    # read txt to pd
    df = pd.read_table(textFile, engine='python', delimiter="|")
    # name
    allName = df.iloc[:, 1]
#    loc_name=allName==targetName
    # item
    allItem = df.columns
    # allItem==item
    loc_item = np.zeros([1, len(allItem)])
    for i in targetItem:
        loc_item = np.vstack([loc_item, allItem == i])
    loc_item = [bool(boo) for boo in np.sum(loc_item, axis=0)]
    content = df.iloc[:, loc_item]
    return content
