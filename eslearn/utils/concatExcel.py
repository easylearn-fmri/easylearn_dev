# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:32:42 2018
此代码用于合并多个excel表格
根据index或者某一列设置为index来合并
@author: lenovo
"""
#import xlrd
import pandas as pd
from pandas import DataFrame
import numpy as np
file = r'D:\myCodes\MVPA_LIChao\MVPA_Python\workstation\0.xlsx'
dataGen = pd.read_excel(file, sheet_name='Sheet1')
dataCell = pd.read_excel(file, sheet_name='Sheet2')
# 'outer':在多个df的index有不同的时候，求他们的并集，区别于left和right
A = dataGen.set_index('folder').join(
    dataCell.set_index('folder'), sort=True, how='left')

A.to_excel('allaa.xlsx')
#All = A.join(dataGen.set_index('folder'),sort=True,how='outer')
# All.to_excel('All.xlsx')

#
data = np.random.randn(4, 3)
frame = DataFrame(data, columns=['year', 'state', 'pop'], index=[1, 2, 3, 4])
data1 = np.random.randn(5, 3)
frame1 = DataFrame(
    data1, columns=[
        'year1', 'state1', 'pop1'], index=[
            2, 3, 4, 5, 1])

All = frame.join(frame1, on=None, how='left', lsuffix='', rsuffix='', sort=True, how='outer')


#
caller = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': [
                      'A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
other = pd.DataFrame(
    {'key': ['K0', 'K1', 'K2', 'K99'], 'B': ['B0', 'B1', 'B2', 'B99']})
caller.join(other.set_index('key'), on='key', how='outer')


##
#wb = xlrd.open_workbook(file)
#
# 获取workbook中所有的表格
#sheets = wb.sheet_names()
#
age_hc = pd.read_excel('age_HC-491.xlsx')
age_hc.to_csv('age_hc.txt', header=False, index=False)

age_sz = pd.read_excel('age_SZ-400.xlsx')
age_sz.to_csv('age_sz.txt', header=False, index=False)

age_hr = pd.read_excel('age_HR-177.xlsx')
age_hr.to_csv('age_hr.txt', header=False, index=False)

#
sex_hc = pd.read_excel('sex_HC-491.xlsx')
sex_hc.to_csv('sex_hc.txt', header=False, index=False)

sex_sz = pd.read_excel('sex_SZ-400.xlsx')
sex_sz.to_csv('sex_sz.txt', header=False, index=False)

sex_hr = pd.read_excel('sex_HR-177.xlsx')
sex_hr.to_csv('sex_hr.txt', header=False, index=False)
#
hc = pd.concat([age_hc, sex_hc], axis=1)
sz = pd.concat([age_sz, sex_sz], axis=1)
hr = pd.concat([age_hr, sex_hr], axis=1)


# dropna
sz = sz.dropna()
refrence = pd.Series(sz.index)

hr = hr.dropna()
refrence = pd.Series(hr.index)

hc = hc.dropna()
refrence = pd.Series(hc.index)
#
hc.to_csv('hc.txt', header=False, index=False, sep=' ')
sz.to_csv('sz.txt', header=False, index=False, sep=' ')
hr.to_csv('hr.txt', header=False, index=False, sep=' ')
