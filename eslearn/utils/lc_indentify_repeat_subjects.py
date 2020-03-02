# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:54:37 2018
找到与subjects重复的ID
@author: lenovo
"""
#
import pandas as pd
from selectSubjID_inScale import loadExcel
from selectSubjID_inScale import selMain


def indentify_repeat_subjects_pairs(file, uid_header):
    """
    Identify the unique ID of subjects that have repeated scan or visit
    """
    allClinicalData = loadExcel(file)
    originSubj = allClinicalData[uid_header]
    folder, basic, hamd17, hama, yars, bprs, logicIndex_scale, logicIndex_repeat = selMain(file)
    dia = allClinicalData[logicIndex_repeat]['诊断备注']
    repeatNote = dia.str.findall(r'(\d*\d)')
    repeatSubj = originSubj.loc[repeatNote.index]
    return repeatNote, repeatSubj


#
if __name__ == '__main__':
    repeatNote, repeatSubj = indentify_repeat_subjects_pairs(
        file=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\10-24大表.xlsx',
        uid_header='folder')
    repeatPairs = pd.concat([repeatSubj, repeatNote], axis=1)
    # 转格式
    repeatPairs = repeatPairs.astype({'folder': 'int'})
    print(repeatPairs)
#    repeatPairs['诊断备注'] = repeatPairs.诊断备注.map(lambda x:float(x))
