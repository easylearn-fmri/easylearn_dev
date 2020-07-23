# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:08:30 2018
1. 筛选age，sex匹配组，并根据age,sex,再次筛选basic
2. 将筛选好的各组被试的folder保存起来（升序排列，如不排列很可能最后相关是对应不上）
3. 根据各组的folder，将fmri数据分别复制或者移动到以组命名的文件夹

被试标签的意义：
1:HC;2:MDD;3:SZ;4:BD;5:HR
@author:LI Chao
"""
# =========================================================================
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from moveScreenedFile import moveMain
import copySelectedFile_OsWalk4 as copy
from selectSubjID_inScale import selMain
import sys
import os
sys.path.append(r'D:\myCodes\MVPA_LIChao\MVPA_Python\workstation')
# 外部导入
#from scipy.stats import chisquare
# ==========================================================================
### input ###

# 是否筛选数据
ifScreen = 1
# 总量表
scaleData = r"D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Scales\8.30大表.xlsx"

# 总数据文件夹
neuroimageDataPath = r'H:\dynamicFC\state\allState17_4\state1'

# 保存量表
ifSave = 0
savePath_scale = r'H:\dynamicFC\state'
if not os.path.exists(savePath_scale):
    os.makedirs(savePath_scale)

# copy or move 总数据到子文件夹
ifCopyOrMove = 0

# copy or move 参数
neuroimageDataPath = r'H:\dynamicFC\state\allState17_4\state4'
savePath_neuroimageData = r'H:\dynamicFC\state\allState17_4'
#
referencePath = [os.path.join(savePath_scale, 'folder_HC.xlsx'),
                 os.path.join(savePath_scale, 'folder_SZ.xlsx'),
                 os.path.join(savePath_scale, 'folder_BD.xlsx'),
                 os.path.join(savePath_scale, 'folder_MDD.xlsx')]

groupName = ['state4_HC', 'state4_SZ', 'state4_BD', 'state4_MDD']

# ==========================================================================

# if ifScreen:


def screen():
    # 筛选量表
    folder, basic, hamd17, hama, yars, bprs, logicIndex_scale, logicIndex_repeat\
        = selMain(scaleData)
    if ifSave:
        folder.to_excel(os.path.join(savePath_scale, 'folder.xlsx'),
                        header=False, index=False)
    # ==========================================================================
    # 求量表被试与数据被试的交集，然后得到筛选的basic
    reguForExtractFileName = r'[1-9]\d*'
    # initiating parameters
    sel = copy.copy_fmri(
        referencePath=os.path.join(
            savePath_scale,
            'folder.xlsx'),
        regularExpressionOfsubjName_forReference='([1-9]\d*)',
        folderNameContainingFile_forSelect='',
        num_countBackwards=1,
        regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',
        keywordThatFileContain='mat',
        neuroimageDataPath=neuroimageDataPath,
        savePath=savePath_scale,
        n_processess=10,
        ifSaveLog=0,
        ifCopy=0,
        ifMove=0,
        saveInToOneOrMoreFolder='saveToOneFolder',
        saveNameSuffix='',
        ifRun=0)
    # run copy or move
    allFilePath, allSubjName, logic_loc,\
        allSelectedFilePath, allSelectedSubjName = sel.main_run()

    # 提取名字。注意：当由多个匹配时，只选择第一个为默认
    allSelectedSubjName = allSelectedSubjName.iloc[:, 0]
    extractedSubjName = allSelectedSubjName.str.findall(reguForExtractFileName)
    extractedSubjName = [extractedSubjName_[0]
                         for extractedSubjName_ in extractedSubjName]
    extractedSubjName = pd.Series(extractedSubjName, dtype='int64')

    # 筛选basic
    basic = basic.set_index('folder').join(
        pd.DataFrame(extractedSubjName).set_index(0),
        sort=True,
        how='inner')  # inner=intersection
    print('### 注意：此时basic的index就是folder名! ###\n')
    # ==========================================================================
    # 根据age再次筛选basic
    ageAll = basic['年龄'].dropna()
    ageLogicInd = (ageAll <= 45) & (ageAll >= 13)
    basic = basic[ageLogicInd]
    # ==========================================================================
    # diagnosis
    dia = basic['诊断']
    dia1Ind = dia == 1
    dia2Ind = dia == 2
    dia3Ind = dia == 3
    dia4Ind = dia == 4
    # ==========================================================================
    # 根据sex再次筛选basic
    # 先求每个组的性别构成
    sex1 = basic['性别'][dia1Ind].dropna()
    sex1Num = np.array([np.sum(sex1 == 1), np.sum(sex1 == 2)])
    sex2 = basic['性别'][dia2Ind].dropna()
    sex2Num = np.array([np.sum(sex2 == 1), np.sum(sex2 == 2)])
    sex3 = basic['性别'][dia3Ind].dropna()
    sex3Num = np.array([np.sum(sex3 == 1), np.sum(sex3 == 2)])
    sex4 = basic['性别'][dia4Ind].dropna()
    sex4Num = np.array([np.sum(sex4 == 1), np.sum(sex4 == 2)])
    # 统计组间性别是否有差异（组1性别1太多）
    # 第一组被试的性别1太多
    sex1Ind1 = sex1[sex1 == 1].index
    sex1Ind2 = sex1[sex1 == 2].index
    screenedsex1Ind1 = sex1Ind1[0:60]  # 只选择前60个，等同于把60以后的筛掉
    sex1 = pd.concat([sex1.loc[screenedsex1Ind1], sex1.loc[sex1Ind2]])
    sex1Num = np.array([np.sum(sex1 == 1), np.sum(sex1 == 2)])
    # ==========================================================================
    # 更新age,并筛选age
    age1 = basic['年龄'].loc[sex1.index]
    age2 = basic['年龄'].loc[sex2.index]
    age3 = basic['年龄'].loc[sex3.index]
    age4 = basic['年龄'].loc[sex4.index]
    # 由于age3偏小，所以筛选掉部分年龄偏小的被试

    def screenAge(age, num):
        # 筛选掉一部分年龄偏小的样本，使各组匹配
        age = age.sort_values()
        screenedAgeInd = age.index[num:]
        screenedAge = age.loc[screenedAgeInd]
        return screenedAge
    age3 = screenAge(age3, 25)
    # ==========================================================================
    # 更新sex
    sex1 = basic['性别'].loc[age1.index]
    sex2 = basic['性别'].loc[age2.index]
    sex3 = basic['性别'].loc[age3.index]
    sex4 = basic['性别'].loc[age4.index]
    sex1Num = np.array([np.sum(sex1 == 1), np.sum(sex1 == 2)])
    sex2Num = np.array([np.sum(sex2 == 1), np.sum(sex2 == 2)])
    sex3Num = np.array([np.sum(sex3 == 1), np.sum(sex3 == 2)])
    sex4Num = np.array([np.sum(sex4 == 1), np.sum(sex4 == 2)])
    # ==========================================================================
    # 根据age或者sex的index,来获得各个诊断的folder,并保存
    folder1 = sex1.index
    folder2 = sex2.index
    folder3 = sex3.index
    folder4 = sex4.index
    # ascending sorted and save
    if ifSave:
        pd.DataFrame(folder1).sort_values(
            by=[0]).to_excel(
            os.path.join(
                savePath_scale,
                'folder_HC.xlsx'),
            header=False,
            index=False)
        pd.DataFrame(folder2).sort_values(
            by=[0]).to_excel(
            os.path.join(
                savePath_scale,
                'folder_MDD.xlsx'),
            header=False,
            index=False)
        pd.DataFrame(folder3).sort_values(
            by=[0]).to_excel(
            os.path.join(
                savePath_scale,
                'folder_SZ.xlsx'),
            header=False,
            index=False)
        pd.DataFrame(folder4).sort_values(
            by=[0]).to_excel(
            os.path.join(
                savePath_scale,
                'folder_BD.xlsx'),
            header=False,
            index=False)

    # ==========================================================================
    # 筛选量表,并保存
    # hamd17
    hamd17_HC = hamd17.loc[folder1].sort_index(axis=0)
    hamd17_MDD = hamd17.loc[folder2].sort_index(axis=0)
    hamd17_SZ = hamd17.loc[folder3].sort_index(axis=0)
    hamd17_BD = hamd17.loc[folder4].sort_index(axis=0)

    if ifSave:
        hamd17_HC = hamd17_HC.to_excel(
            os.path.join(
                savePath_scale,
                'hamd17_HC.xlsx'))
        hamd17_MDD = hamd17_MDD.to_excel(
            os.path.join(savePath_scale, 'hamd17_MDD.xlsx'))
        hamd17_SZ = hamd17_SZ.to_excel(
            os.path.join(
                savePath_scale,
                'hamd17_SZ.xlsx'))
        hamd17_BD = hamd17_BD.to_excel(
            os.path.join(
                savePath_scale,
                'hamd17_BD.xlsx'))

    # hama
    hama_HC = hama.loc[folder1].sort_index(axis=0)
    hama_MDD = hama.loc[folder2].sort_index(axis=0)
    hama_SZ = hama.loc[folder3].sort_index(axis=0)
    hama_BD = hama.loc[folder4].sort_index(axis=0)

    if ifSave:
        hama_HC = hama_HC.to_excel(
            os.path.join(
                savePath_scale,
                'hama_HC.xlsx'))
        hama_MDD = hama_MDD.to_excel(
            os.path.join(
                savePath_scale,
                'hama_MDD.xlsx'))
        hama_SZ = hama_SZ.to_excel(
            os.path.join(
                savePath_scale,
                'hama_SZ.xlsx'))
        hama_BD = hama_BD.to_excel(
            os.path.join(
                savePath_scale,
                'hama_BD.xlsx'))

    # yars
    yars_HC = yars.loc[folder1].sort_index(axis=0)
    yars_MDD = yars.loc[folder2].sort_index(axis=0)
    yars_SZ = yars.loc[folder3].sort_index(axis=0)
    yars_BD = yars.loc[folder4].sort_index(axis=0)

    if ifSave:
        yars_HC = yars_HC.to_excel(
            os.path.join(
                savePath_scale,
                'yars_HC.xlsx'))
        yars_MDD = yars_MDD.to_excel(
            os.path.join(
                savePath_scale,
                'yars_MDD.xlsx'))
        yars_SZ = yars_SZ.to_excel(
            os.path.join(
                savePath_scale,
                'yars_SZ.xlsx'))
        yars_BD = yars_BD.to_excel(
            os.path.join(
                savePath_scale,
                'yars_BD.xlsx'))

    # bprs
    bprs_HC = bprs.loc[folder1].sort_index(axis=0)
    bprs_MDD = bprs.loc[folder2].sort_index(axis=0)
    bprs_SZ = bprs.loc[folder3].sort_index(axis=0)
    bprs_BD = bprs.loc[folder4].sort_index(axis=0)

    if ifSave:
        bprs_HC = bprs_HC.to_excel(
            os.path.join(
                savePath_scale,
                'bprs_HC.xlsx'))
        bprs_MDD = bprs_MDD.to_excel(
            os.path.join(
                savePath_scale,
                'bprs_MDD.xlsx'))
        bprs_SZ = bprs_SZ.to_excel(
            os.path.join(
                savePath_scale,
                'bprs_SZ.xlsx'))
        bprs_BD = bprs_BD.to_excel(
            os.path.join(
                savePath_scale,
                'bprs_BD.xlsx'))

    # ==========================================================================
    # save to excel
    # join
    scale_HC = pd.DataFrame(age1).join(pd.DataFrame(sex1), sort=True)
    scale_MDD = pd.DataFrame(age2).join(pd.DataFrame(sex2), sort=True)
    scale_SZ = pd.DataFrame(age3).join(pd.DataFrame(sex3), sort=True)
    scale_BD = pd.DataFrame(age4).join(pd.DataFrame(sex4), sort=True)
    # save
    if ifSave:
        scale_HC.to_excel(os.path.join(savePath_scale, 'ageANDsex_HC.xlsx'))
        scale_MDD.to_excel(os.path.join(savePath_scale, 'ageANDsex_MDD.xlsx'))
        scale_SZ.to_excel(os.path.join(savePath_scale, 'ageANDsex_SZ.xlsx'))
        scale_BD.to_excel(os.path.join(savePath_scale, 'ageANDsex_BD.xlsx'))

        scale_HC.to_csv(
            os.path.join(
                savePath_scale,
                'ageANDsex_HC.txt'),
            header=0,
            index=False,
            sep=' ')
        scale_MDD.to_csv(
            os.path.join(
                savePath_scale,
                'ageANDsex_MDD.txt'),
            header=0,
            index=False,
            sep=' ')
        scale_SZ.to_csv(
            os.path.join(
                savePath_scale,
                'ageANDsex_SZ.txt'),
            header=0,
            index=False,
            sep=' ')
        scale_BD.to_csv(
            os.path.join(
                savePath_scale,
                'ageANDsex_BD.txt'),
            header=0,
            index=False,
            sep=' ')

# ==========================================================================
# 统计
# age
#f,p = f_oneway(age1,age2,age3,age4)
# ==========================================================================

# ==========================================================================
# 找到有完整量表数据的被试
# bprs_SZ1=bprs_SZ.dropna()
# folder_SZ1=pd.Series(bprs_SZ1.index)
# ===========================================================================
# if ifCopyOrMove:


def copyOrMove():
    # 将某些被试复制到其他地方

    #    neuroimageDataPath=r'H:\dynamicFC\state\allState17_4\state1'
    #    savePath_neuroimageData=r'H:\dynamicFC\state\allState17_4\state1_HC'
    #    #
    #    referencePath=[os.path.join(savePath_scale,'folder_HC.xlsx'),
    #                   os.path.join(savePath_scale,'folder_SZ.xlsx'),
    #                   os.path.join(savePath_scale,'folder_BD.xlsx'),
    #                   os.path.join(savePath_scale,'folder_MDD.xlsx')]
    #    #
    #    groupName=['HC','SZ','BD','MDD']

    #
    import copySelectedFile_OsWalk4 as copy
    for (referencepath, groupname) in zip(referencePath, groupName):
        # initiating parameters
        sel = copy.copy_fmri(
            referencePath=referencepath,
            regularExpressionOfsubjName_forReference='([1-9]\d*)',
            folderNameContainingFile_forSelect='',
            num_countBackwards=1,
            regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',
            keywordThatFileContain='',
            neuroimageDataPath=neuroimageDataPath,
            savePath=os.path.join(
                savePath_neuroimageData,
                groupname),
            n_processess=10,
            ifSaveLog=0,
            ifCopy=0,
            ifMove=1,
            saveInToOneOrMoreFolder='saveToOneFolder',
            saveNameSuffix='',
            ifRun=1)
        # run copy or move
        allFilePath, allSubjName, logic_loc,\
            allSelectedFilePath, allSelectedSubjName = sel.main_run()
    #
    print('Done!')


# ==========================================================================
if __name__ == '__main__':
    # screen
    if ifScreen:
        screen()

    # copy or move
    if ifCopyOrMove:
        copyOrMove()
