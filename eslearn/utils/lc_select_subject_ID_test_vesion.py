# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:36:04 2018

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 21:53:10 2018
改自selectSubjID_inScale_V2
筛选大表的item和subjects
最后得到诊断为[1,2,3,4]，扫描质量良好，而且不重复的被试

inputs:
        file_all：大表

        column_basic1=[0,11,19,20,21,22,23,27,28,29,30]：基本信息列
        column_basic2=['学历（年）','中国人利手量表']：基本信息名
        column_hamd17=np.arange(104,126,1),
        column_hama=np.arange(126,141,1),
        column_yars=np.arange(141,153,1),
        column_bprs=np.arange(153,177,1)

        column_diagnosis='诊断':诊断的列名
        column_quality='Resting_quality'
        column_note1='诊断备注'
        column_note2='备注'
        note1_keyword='复扫':重复备注文字

outputs:
        folder:筛选出来的ID
        basic:筛选出来的基本信息
        hamd17,hamm,yars,bprs:筛选出来的量表
        logicIndex_scale:量表的逻辑index
        logicIndex_repeat:重复量表的index
@author: li Chao
new feature:不限定条件列以及筛选条件
"""
# ===============================================
#import re




import pandas as pd
import os
import numpy as np
class select_SubjID():
    # initial parameters
    def __init__(self,

                 file_all=r"D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Scales\8.30大表.xlsx",

                 # 基本信息和量表，暂时不能加条件来筛选行
                 column_basic1=[0, 11, 19, 20, 21, 22, 23, 27, 28, 29, 30],
                 column_basic2=['学历（年）', '中国人利手量表'],
                 column_hamd17=np.arange(104, 126, 1),
                 column_hama=np.arange(126, 141, 1),
                 column_yars=np.arange(141, 153, 1),
                 column_bprs=np.arange(153, 177, 1),

                 # 可以加条件筛选行的列（item）,字典形式，key为列名，value为条件
                 # condition_name:{condition:[include_or_exclude,match_method]}
                 screening_dict={
                     '诊断': {1: ['include', 'exact'], 2: ['include', 'exact'], 3: ['include', 'exact'], 4: ['include', 'exact']},
                     'Resting_quality': {'Y': ['include', 'exact']},
                     '诊断备注': {'复扫': ['exclude', 'fuzzy'], '糖尿病': ['exclude', 'fuzzy'], '不能入组': ['exclude', 'fuzzy']},
                     '备注': {'复扫': ['exclude', 'fuzzy']}
                 }

                 #                screening_dict={
                 #                                '诊断':{1:['include','exact'],2:['include','exact'],3:['include','exact'],4:['include','exact']},
                 #                                'Resting_quality':{'Y':['include','exact']},
                 #                                '诊断备注':{'复扫':['exclude','fuzzy']}
                 #                                }
                 ):
        # ============================================

        self.file_all = file_all

        self.column_basic1 = column_basic1
        self.column_basic2 = column_basic2
        self.column_hamd17 = column_hamd17
        self.column_hama = column_hama
        self.column_yars = column_yars
        self.column_bprs = column_bprs

        self.screening_dict = screening_dict

        print('Initialized!\n')

    # ====================================================

    def loadExcel(self):
        # load all clinical data in excel
        self.allClinicalData = pd.read_excel(self.file_all)
        return self

    def extract_one_series(self, column_var):
        # 选项目，项目列可以是数字编号，也可以是列名字符串
        if isinstance(column_var[0], str):
            data = self.allClinicalData.loc[:, column_var]
        elif isinstance(self.column_basic1[0], np.int32):
            data = self.allClinicalData.iloc[:, column_var]
        elif isinstance(self.column_basic1[0], int):
            data = self.allClinicalData.iloc[:, column_var]
        else:
            print('basicIndex 的输入有误！\n')
        return data
    # ====================================================

    def select_item(self):
        # 选项目，项目列可以是数字编号，也可以是列名字符串（注意：这些项目暂时不支持行筛选）
        basic1 = self.extract_one_series(self.column_basic1)
        basic2 = self.extract_one_series(self.column_basic2)
        self.basic = pd.concat([basic1, basic2], axis=1)

        self.hamd17 = self.extract_one_series(self.column_hamd17)

        self.hama = self.extract_one_series(self.column_hama)

        self.yars = self.extract_one_series(self.column_yars)

        self.bprs = self.extract_one_series(self.column_bprs)

        return self

    # ====================================================
    # 条件筛选

    def screen_data_according_conditions_in_dict_one(
            self, series_for_screening, condition_in_dict):
        # 根据字典里面的条件筛选，并得到index。注意条件可能是字符串也可以是数字。
        # 注意：此函数只处理一列。
        # 第一种情况：是字符串的情况。

        # 由于contains函数不能处理null，先把null替换为'未知'
        series_for_screening = series_for_screening.mask(
            series_for_screening.isnull(), '未知')

        # 生成index为series_for_screening的index的空pd.DataFrame,用于后续join
        screened_ind_all = pd.DataFrame([])
        for condition_name in condition_in_dict:
            screened_ind = pd.DataFrame([], index=series_for_screening.index)
            # 每个key值筛选后，都用pd.DataFrame.join方法求并集
            print(condition_name)
            print(condition_in_dict[condition_name])
            print(condition_in_dict[condition_name][-1])

            # 进入条件筛选
            # 精确匹配,一般数字为精确匹配
            if condition_in_dict[condition_name][-1] == 'exact':
                if condition_in_dict[condition_name][0] == 'exclude':
                    screened_ind = screened_ind.loc[series_for_screening.index[series_for_screening != condition_name]]
                elif condition_in_dict[condition_name][0] == 'include':
                    screened_ind = screened_ind.loc[series_for_screening.index[series_for_screening == condition_name]]

             # 模糊匹配
            elif condition_in_dict[condition_name][-1] == 'fuzzy':
                if condition_in_dict[condition_name][0] == 'exclude':
                    screened_ind_tmp = series_for_screening.mask(
                        series_for_screening.str.contains(condition_name), None).dropna()
                    screened_ind = screened_ind.loc[screened_ind_tmp.dropna(
                    ).index]
                elif condition_in_dict[condition_name][0] == 'include':
                    screened_ind_tmp = series_for_screening.where(
                        series_for_screening.str.contains(condition_name), None)
                    screened_ind = screened_ind.loc[screened_ind_tmp.dropna(
                    ).index]

            # 未指名匹配方式
            else:
                print(
                    '__ini__ is wrong!\n### may be words "exact OR fuzzy" is wrong ###\n')

            # pd.join 求并集或交集
            if screened_ind_all.index.empty:
                screened_ind_all = screened_ind_all.join(
                    pd.DataFrame(screened_ind), how='outer')
            else:
                if condition_in_dict[condition_name][0] == 'exclude':
                    screened_ind_all = screened_ind_all.join(
                        pd.DataFrame(screened_ind), how='inner')
                elif condition_in_dict[condition_name][0] == 'include':
                    screened_ind_all = screened_ind_all.join(
                        pd.DataFrame(screened_ind), how='outer')
        return screened_ind_all

    def screen_data_according_conditions_in_dict_all(self):

        # 把字典中每一列的index筛选出来，然后逐个join求交集，从而得到满足所有条件的index

        index_selected = pd.DataFrame([], index=self.allClinicalData.index)
        for key in self.screening_dict:
            #            key='诊断备注'
            condition_in_dict = self.screening_dict[key]
            series_for_screening = self.extract_one_series(key)
            screened_ind_all = self.screen_data_according_conditions_in_dict_one(
                series_for_screening, condition_in_dict)

            # join 交集
            self.index_selected = index_selected.join(
                screened_ind_all, how='inner')
        return self

    # ====================================================

    def selcet_subscale_according_index(self):
        # 根据index 选择量表
        self.basic = self.basic.loc[self.index_selected.index]
        self.hamd17 = self.hamd17.loc[self.index_selected.index]
        self.hama = self.hama.loc[self.index_selected.index]
        self.yars = self.yars.loc[self.index_selected.index]
        self.bprs = self.bprs.loc[self.index_selected.index]
#        print('bprs2:{}\n'.format(self.bprs))
        return self
    # ===============================================

        print('Running...\n')
        # load
        self = self.loadExcel()    # item

        # item
        self = self.select_item()

        # 条件筛选
        self.screen_data_according_conditions_in_dict_all()

        # 亚量表
        self = self.selcet_subscale_according_index()

        # folder ID
        self.folder = self.basic['folder']

        #
        print('Done!\n')
        return self


# ===============================================
if __name__ == '__main__':
    #    print('==================================我是分割线====================================\n')
    # allFile=r"D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Scales\8.30大表.xlsx"
    import select_subject_ID_test_vesion as select
#
#    current_path=os.getcwd()
#    print('当前路径是：[{}]\n'.format(current_path))
#
#    ini_path=os.path.join(current_path,'__ini__.txt')
#    print('初始化参数位于：[{}]\n'.format(ini_path))
#
#    print('正在读取初始化参数...\n')
#    ini = open(ini_path).read()
#    ini=ini.strip('').split('\n')
#    ini=[ini_ for ini_ in ini if ini_.strip()]
#
#    name=locals()
#    for ini_param in ini:
#        name[ini_param.strip().split('=')[0]]=eval(ini_param.strip().split('=')[1])
#
#        print('{}={}\n'.format(ini_param.strip().split('=')[0],name[ini_param.strip().split('=')[0]]))
#    print('初始化参数读取完成!\n')
    self = select.select_SubjID()
#    self=select.select_SubjID(
#                            file_all=file_all,
#
#                            column_basic1=column_basic1,
#                            column_basic2=column_basic2,
#                            column_hamd17=column_hamd17,
#                            column_hama=column_hama,
#                            column_yars=column_yars,
#                            column_bprs=column_bprs,
#
#                            column_diagnosis=column_diagnosis,
#                            diagnosis_label=diagnosis_label,
#
#                            column_quality=column_quality,
#                            quality_keyword=quality_keyword,
#
#                            column_note1=column_note1,
#                            note1_keyword=note1_keyword,
#
#                            column_note2=column_note2,
#                            note2_keyword=note2_keyword
#                            )
#
#    results=self.selMain()
#
#    # check results
#    results_dict=results.__dict__
#    print('所有结果为:{}\n'.format(list(results_dict.keys())))
#
#    results.folder.to_excel('folder.xlsx',header=False,index=False)
#    print('###筛选的folder 保存在:[{}]###\n'.format(os.path.join(current_path,'folder.xlsx')))
#    print('作者：黎超\n邮箱：lichao19870617@163.com\n')
#    input("######按任意键推出######\n")
#    print('==================================我是分割线====================================\n')
#
#
# a=results_dict.keys()
# a=list(a)
# print(results_dict.keys())
##import codecs
##data = open(ini).read()
# if data[:3] == codecs.BOM_UTF8:
##	data = data[3:]
# print data.decode("utf-8")
##
# basic1=self.allClinicalData[['诊断','备注']]
