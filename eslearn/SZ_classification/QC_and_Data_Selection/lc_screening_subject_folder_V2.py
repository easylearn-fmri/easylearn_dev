# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 20:27:50 2018
筛选大表的item和subjects
最后得到诊断为[1,2,3,4]，扫描质量良好，而且不重复的被试

inputs:
        file_all：大表

        basicIndex_iloc=[0,11,19,20,21,22,23,27,28,29,30]：基本信息列
        basicIndex_str=['学历（年）','中国人利手量表']：基本信息名
        hamd17Index_iloc=np.arange(104,126,1),
        hamaIndex_iloc=np.arange(126,141,1),
        yarsIndex_iloc=np.arange(141,153,1),
        bprsIndex_iloc=np.arange(153,177,1)

        diagnosis_column_name='诊断':诊断的列名
        quality_column_name='Resting_quality'
        note1_column_name='诊断备注'
        note2_column_name='备注'
        note1_keyword='复扫':重复备注文字

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
import os
import numpy as np


class select_SubjID():
    # initial parameters
    def __init__(self,

                 file_all=r'..\大表.xlsx',

                 basicIndex_iloc=[0, 11, 19, 20, 21, 22, 23, 27, 28, 29, 30],
                 basicIndex_str=['学历（年）', '中国人利手量表'],
                 hamd17Index_iloc=np.arange(104, 126, 1),
                 hamaIndex_iloc=np.arange(126, 141, 1),
                 yarsIndex_iloc=np.arange(141, 153, 1),
                 bprsIndex_iloc=np.arange(153, 177, 1),

                 diagnosis_column_name='诊断',
                 diagnosis_label=[1, 2, 3, 4],

                 quality_column_name='Resting_quality',
                 quality_keyword='Y',

                 note1_column_name='诊断备注',
                 note1_keyword='复扫',

                 note2_column_name='备注',
                 note2_keyword='复扫'):
           # ======================================
        self.file_all = file_all

        self.basicIndex_iloc = basicIndex_iloc
        self.basicIndex_str = basicIndex_str
        self.hamd17Index_iloc = hamd17Index_iloc
        self.hamaIndex_iloc = hamaIndex_iloc
        self.yarsIndex_iloc = yarsIndex_iloc
        self.bprsIndex_iloc = bprsIndex_iloc

        self.diagnosis_column_name = diagnosis_column_name
        self.diagnosis_label = diagnosis_label
        self.quality_column_name = quality_column_name
        self.quality_keyword = quality_keyword

        self.note1_column_name = note1_column_name
        self.note1_keyword = note1_keyword
        self.note2_column_name = note2_column_name
        self.note2_keyword = note2_keyword
        print('Initialized!\n')

    # ====================================================

    def loadExcel(self):
        # load all clinical data in excel
        self.allClinicalData = pd.read_excel(self.file_all)
        return self
    # ===============================================

    def select_item(self):
        #        ini=np.int32(1)
        # 选项目
        if isinstance(self.basicIndex_iloc[0], str):
            basic1 = self.allClinicalData.loc[:, self.basicIndex_iloc]
        elif isinstance(self.basicIndex_iloc[0], np.int32):
            basic1 = self.allClinicalData.iloc[:, self.basicIndex_iloc]
        elif isinstance(self.basicIndex_iloc[0], int):
            basic1 = self.allClinicalData.iloc[:, self.basicIndex_iloc]
        else:
            print('basicIndex 的输入有误！\n')
        basic2 = self.allClinicalData[self.basicIndex_str]
        self.basic = pd.concat([basic1, basic2], axis=1)

        if isinstance(self.hamd17Index_iloc[0], str):
            self.hamd17 = self.allClinicalData.loc[:, self.hamd17Index_iloc]
        elif isinstance(self.hamd17Index_iloc[0], int):
            self.hamd17 = self.allClinicalData.iloc[:, self.hamd17Index_iloc]
        elif isinstance(self.hamd17Index_iloc[0], np.int32):
            self.hamd17 = self.allClinicalData.iloc[:, self.hamd17Index_iloc]
        else:
            print('hamd17Index_iloc 的输入有误！\n')

        if isinstance(self.hamaIndex_iloc[0], str):
            self.hama = self.allClinicalData.loc[:, self.hamaIndex_iloc]
        elif isinstance(self.hamaIndex_iloc[0], int):
            self.hama = self.allClinicalData.iloc[:, self.hamaIndex_iloc]
        elif isinstance(self.hamaIndex_iloc[0], np.int32):
            self.hama = self.allClinicalData.iloc[:, self.hamaIndex_iloc]
        else:
            print('hamaIndex_iloc 的输入有误！\n')

        if isinstance(self.yarsIndex_iloc[0], str):
            self.yars = self.allClinicalData.loc[:, self.yarsIndex_iloc]
        elif isinstance(self.yarsIndex_iloc[0], int):
            self.yars = self.allClinicalData.iloc[:, self.yarsIndex_iloc]
        elif isinstance(self.yarsIndex_iloc[0], np.int32):
            self.yars = self.allClinicalData.iloc[:, self.yarsIndex_iloc]
        else:
            print('yarsIndex_iloc 的输入有误！\n')

        if isinstance(self.bprsIndex_iloc[0], str):
            self.bprs = self.allClinicalData.loc[:, self.bprsIndex_iloc]
        elif isinstance(self.bprsIndex_iloc[0], int):
            self.bprs = self.allClinicalData.iloc[:, self.bprsIndex_iloc]
        elif isinstance(self.bprsIndex_iloc[0], np.int32):
            self.bprs = self.allClinicalData.iloc[:, self.bprsIndex_iloc]
        else:
            print('bprsIndex_iloc 的输入有误！\n')

#        print('bprs1:{}\n'.format(self.bprs))
        return self
    # ===============================================

    def select_diagnosis(self):
        # 诊断
        diagnosis = self.allClinicalData[self.diagnosis_column_name]

        logicIndex_diagnosis = pd.DataFrame(
            np.ones([len(self.allClinicalData), 1]) == 0).iloc[:, 0]

        for i, dia in enumerate(self.diagnosis_label):

            dia = diagnosis.loc[:] == dia

            logicIndex_diagnosis = pd.Series(
                logicIndex_diagnosis.values | dia.values)

#            logicIndex_diagnosis=(diagnosis==1 )|(diagnosis==2 )|(diagnosis==3)|(diagnosis==4)
        self.ind_diagnosis = logicIndex_diagnosis.index[logicIndex_diagnosis]
        return self

    def select_quality(self):
        # 根据resting质量筛选样本
        logicIndex_quality = self.allClinicalData[self.quality_column_name] == self.quality_keyword
        self.ind_quality = logicIndex_quality.index[logicIndex_quality]
        return self

    def select_note1(self):
        # 检查复扫,筛选
        #   note1_keyword=r'.*?复扫.*?'
        note1_column_name = self.allClinicalData[self.note1_column_name]
        note1_column_name = note1_column_name.where(
            note1_column_name.notnull(), '未知')
        index_repeat = note1_column_name.str.contains(self.note1_keyword)
        logicIndex_repeat = [bool(index_repeat_)
                             for index_repeat_ in index_repeat]
        logicIndex_notRepeat = [index_repeat_ ==
                                0 for index_repeat_ in index_repeat]

        self.ind_note1 = self.allClinicalData.index[logicIndex_repeat]
        self.ind_not_note1 = self.allClinicalData.index[logicIndex_notRepeat]

        return self

    def select_note2(self):
        # 检查复扫,筛选
        #   note1_keyword=r'.*?复扫.*?'
        note2_column_name = self.allClinicalData[self.note2_column_name]
        note2_column_name = note2_column_name.where(
            note2_column_name.notnull(), '未知')
        index_repeat = note2_column_name.str.contains(self.note2_keyword)
        logicIndex_repeat = [bool(index_repeat_)
                             for index_repeat_ in index_repeat]
        logicIndex_notRepeat = [index_repeat_ ==
                                0 for index_repeat_ in index_repeat]

        self.ind_note2 = self.allClinicalData.index[logicIndex_repeat]
        self.ind_not_note2 = self.allClinicalData.index[logicIndex_notRepeat]

        return self

    def select_intersection(self):
        # index 交集
        # 诊断*扫描质量
        self.ind_selected = pd.DataFrame(
            self.ind_diagnosis).set_index(0).join(
            pd.DataFrame(
                self.ind_quality).set_index(0),
            how='inner')

        # 诊断*扫描质量*note1_column_name
        self.ind_selected = pd.DataFrame(
            self.ind_selected).join(
            pd.DataFrame(
                self.ind_not_note1).set_index(0),
            how='inner')

        # 诊断*扫描质量*note1_column_name*note2_column_name
        self.ind_selected = pd.DataFrame(
            self.ind_selected).join(
            pd.DataFrame(
                self.ind_not_note2).set_index(0),
            how='inner')
#
        # 筛选folder
        self.folder = self.allClinicalData['folder'].loc[self.ind_selected.index]

        return self
    # ===============================================

    def selcet_subscale_according_index(self):
        # 根据index 选择量表
        self.basic = self.basic.loc[self.ind_selected.index]
        self.hamd17 = self.hamd17.loc[self.ind_selected.index]
        self.hama = self.hama.loc[self.ind_selected.index]
        self.yars = self.yars.loc[self.ind_selected.index]
        self.bprs = self.bprs.loc[self.ind_selected.index]
#        print('bprs2:{}\n'.format(self.bprs))
        return self
# =============================================================================
#     def dropnan(self,scale):
#         # 把量表中的空缺去除
#         nanIndex_scale=scale.isnull()
#         nanIndex_scale=np.sum(nanIndex_scale.values,axis=1)
#         logicIndex_scale=nanIndex_scale==0
#         return logicIndex_scale
#
#     def dropnan_all(self,scale):
#         logicIndex_scale=[dropnan(scale_) for scale_ in scale]
#         return logicIndex_scale
# =============================================================================
    # ===============================================

    def selMain(self):
        print('Running...\n')
        # load
        self = self.loadExcel()    # item

        self = self.select_item()

        # diagnosis_column_name
        self = self.select_diagnosis()

        # quality_column_name
        self = self.select_quality()

        # repeat
        self = self.select_note1()
        self = self.select_note2()

        # intersection
        self = self.select_intersection()

        # select
        self = self.selcet_subscale_according_index()

        # folder ID
#        folder=basic['folder']
#        # drop nan
#        logicIndex_scale=\
#        dropnan_all(scale=[hamd17,hama,yars,bprs])
#        print(self.folder)
        print('Done!\n')
        return self


# ===============================================
if __name__ == '__main__':
    print('==================================我是分割线====================================\n')
#    allFile=r"D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Scales\8.30大表.xlsx"
    import selectSubjID_inScale_V2 as select

    current_path = os.getcwd()
    print('当前路径是：[{}]\n'.format(current_path))

    ini_path = os.path.join(current_path, '__ini__.txt')
    print('初始化参数位于：[{}]\n'.format(ini_path))

    print('正在读取初始化参数...\n')
    ini = open(ini_path).read()
    ini = ini.strip('').split('\n')
    ini = [ini_ for ini_ in ini if ini_.strip()]

    name = locals()
    for ini_param in ini:
        name[ini_param.strip().split('=')[0]] = eval(
            ini_param.strip().split('=')[1])

        print('{}={}\n'.format(ini_param.strip().split('=')
                               [0], name[ini_param.strip().split('=')[0]]))
    print('初始化参数读取完成!\n')

    sel = select.select_SubjID(
        file_all=file_all,

        basicIndex_iloc=basicIndex_iloc,
        basicIndex_str=basicIndex_str,
        hamd17Index_iloc=hamd17Index_iloc,
        hamaIndex_iloc=hamaIndex_iloc,
        yarsIndex_iloc=yarsIndex_iloc,
        bprsIndex_iloc=bprsIndex_iloc,

        diagnosis_column_name=diagnosis_column_name,
        diagnosis_label=diagnosis_label,

        quality_column_name=quality_column_name,
        quality_keyword=quality_keyword,

        note1_column_name=note1_column_name,
        note1_keyword=note1_keyword,

        note2_column_name=note2_column_name,
        note2_keyword=note2_keyword)

    results = sel.selMain()

    # check results
    results_dict = results.__dict__
    print('所有结果为:{}\n'.format(list(results_dict.keys())))

    results.folder.to_excel('folder.xlsx', header=False, index=False)
    print(
        '###筛选的folder 保存在:[{}]###\n'.format(
            os.path.join(
                current_path,
                'folder.xlsx')))
    print('作者：黎超\n邮箱：lichao19870617@163.com\n')
    input("######按任意键推出######\n")
    print('==================================我是分割线====================================\n')
