# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:43:28 2018
筛选大表的item和subjects
最后得到诊断为[1,2,3,4]，扫描质量良好，而且不重复的被试
outputs:
        folder:筛选出来的ID
        basic:筛选出来的基本信息
        hamd17,hamm,yars,bprs:筛选出来的量表
        logicIndex_scale:量表的逻辑index
        logicIndex_repeat:重复量表的index
@author: lenovo
"""
# ===============================================
import pandas as pd
#import re
import numpy as np


def loadExcel(file_all):
    # load all clinical data in excel
    allClinicalData = pd.read_excel(file_all)
    return allClinicalData
# ===============================================


def select_item(allClinicalData,
                basicIndex_iloc=[0, 11, 19, 20, 21, 22, 23, 27, 28, 29, 30],
                basicIndex_str=['学历（年）', '中国人利手量表'],
                hamd12Index_iloc=np.arange(104, 126, 1),
                hamaIndex_iloc=np.arange(126, 141, 1),
                yarsIndex_iloc=np.arange(141, 153, 1),
                bprsIndex_iloc=np.arange(153, 177, 1)):
    # 选项目
    basic1 = allClinicalData.iloc[:, basicIndex_iloc]
    basic2 = allClinicalData[basicIndex_str]
    basic = pd.concat([basic1, basic2], axis=1)
    hamd17 = allClinicalData.iloc[:, hamd12Index_iloc]
    hama = allClinicalData.iloc[:, hamaIndex_iloc]
    yars = allClinicalData.iloc[:, yarsIndex_iloc]
    bprs = allClinicalData.iloc[:, bprsIndex_iloc]
    return basic, hamd17, hama, yars, bprs
# ===============================================


def diagnosis(diagnosis):
    # 诊断
    logicIndex_diagnosis = (
        diagnosis == 1) | (
        diagnosis == 2) | (
            diagnosis == 3) | (
                diagnosis == 4)
    return logicIndex_diagnosis


def select_quality(quality):
    # 根据resting质量筛选样本
    logicIndex_quality = quality == 'Y'
    return logicIndex_quality


def select_repeat(note, repeatMarker='复扫'):
    # 检查复扫,筛选
    #   repeatMarker=r'.*?复扫.*?'
    note = note.where(note.notnull(), '未知')
    index_repeat = note.str.contains(repeatMarker)
    logicIndex_repeat = [bool(index_repeat_) for index_repeat_ in index_repeat]
    logicIndex_notRepeat = [index_repeat_ == 0 for index_repeat_ in index_repeat]
    return logicIndex_repeat, logicIndex_notRepeat


def select_intersection(
        logicIndex_diagnosis,
        logicIndex_quality,
        logicIndex_notRepeat):
    # index 交集
    index_seleled = logicIndex_diagnosis & logicIndex_quality & logicIndex_notRepeat
    return index_seleled
# ===============================================


def selcetSubj_accordingLogicIndex(index_seleled,
                                   basic, hamd17, hama, yars, bprs):
    # 根据index 选择量表
    basic = basic[index_seleled]
    hamd17 = hamd17[index_seleled]
    hama = hama[index_seleled]
    yars = yars[index_seleled]
    bprs = bprs[index_seleled]
    return basic, hamd17, hama, yars, bprs
# ===============================================


def dropnan(scale):
    # 把量表中的空缺去除
    nanIndex_scale = scale.isnull()
    nanIndex_scale = np.sum(nanIndex_scale.values, axis=1)
    logicIndex_scale = nanIndex_scale == 0
    return logicIndex_scale


def dropnan_all(scale):
    logicIndex_scale = [dropnan(scale_) for scale_ in scale]
    return logicIndex_scale
# ===============================================


def selMain(allFile):
    # load
    allClinicalData = loadExcel(allFile)    # item
    basic, hamd17, hama, yars, bprs = select_item(
        allClinicalData, basicIndex_iloc=[
            0, 11, 19, 20, 21, 22, 23, 27, 28, 29, 30], basicIndex_str=[
            '学历（年）', '中国人利手量表'], hamd12Index_iloc=np.arange(
                104, 126, 1), hamaIndex_iloc=np.arange(
                    126, 141, 1), yarsIndex_iloc=np.arange(
                        141, 153, 1), bprsIndex_iloc=np.arange(
                            153, 177, 1))

    # diagnosis
    logicIndex_diagnosis = diagnosis(diagnosis=basic['诊断'])
    # quality
    logicIndex_quality = select_quality(quality=basic['Resting_quality'])
    # repeat
    logicIndex_repeat, logicIndex_notRepeat =select_repeat(note=basic['诊断备注'], repeatMarker='复扫')
    # intersection
    index_seleled = select_intersection(
        logicIndex_diagnosis,
        logicIndex_quality,
        logicIndex_notRepeat)
    # select
    basic, hamd17, hama, yars, bprs =\
        selcetSubj_accordingLogicIndex(index_seleled,
                                       basic, hamd17, hama, yars, bprs)
    # folder ID
    folder = basic['folder']
    # drop nan
    logicIndex_scale =\
        dropnan_all(scale=[hamd17, hama, yars, bprs])
    return folder, basic, hamd17, hama, yars, bprs, logicIndex_scale, logicIndex_repeat


# ===============================================
if __name__ == '__main__':
    file_all = r"D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\10-24大表.xlsx"
    folder, basic, hamd17, hama, yars, bprs, logicIndex_scale, logicIndex_repeat = selMain(file_all)
    # save
#    folder_to_save=pd.DataFrame(folder)
#    folder_to_save.to_excel(r"D:\LI_Chao_important_don't_delete\Data\workstation_20180829_dynamicFC\folder.xlsx")
#    a=pd.read_excel(r'D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\folder.xlsx')
#    a=a['folder']
    print(folder)
    print('Done!')
