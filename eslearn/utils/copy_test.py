# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:05:28 2018:
在版本3的基础上,根据pandas的join方法来求交集

根据从量表中筛选的样本，来获得符合要求的原始数据的路径
数据结构neuroimageDataPath//subject00001//files
也可以是任何的数据结构，只要给定subjName在哪里就行
总之，最后把file复制到其他地方（可以限定某个file）

input:
    #1   referencePath:需要复制的被试名字所在text文件（大表中的folder）
    #2   regularExpressionOfsubjName_forReference:如提取量表中subjName的正则表达式
        ith: 量表中的subjName有多个匹配项时，选择第几个
    #3   folderNameContainingFile_forSelect:想把被试的哪个模态/或那个文件夹下的文件复制出来（如同时有'resting'和'dti'时，选择那个模态）
    #4   num_countBackwards:subjName在倒数第几个block内(第一个计数为1)
    #   如'D:\myCodes\workstation_20180829_dynamicFC\FunImgARW\1-500\00002_resting\dti\dic.txt'
    #  的subjName在倒数第3个中
    #5   regularExpressionOfSubjName_forNeuroimageDataFiles:用来筛选mri数据中subject name字符串的正则表达式
        ith_subjName: 当subject name中有多个字符串匹配时，选择第几个（默认第一个）
    #6   keywordThatFileContain:用来筛选file的正则表达式或keyword
    #7   neuroimageDataPath：原始数据的根目录
    #8   savePath: 将原始数据copy到哪个大路径
    #    n_processess=5几个线程
    #9  ifSaveLog：是否保存复制log
    #10  ifCopy：是否执行复制功能
    #11 ifMove:是否移动（0）
    #12  saveInToOneOrMoreFolder：保存到每个被试文件夹下，还是保存到一个文件夹下
    #13  saveNameSuffix：文件保存的尾缀（'.nii'）
    #14  ifRun:是否真正对文件执行移动或复制（0）
    #   总体来说被复制的文件放在如下的路径：savePath/saveFolderName/subjName/files
@author: LI Chao
new featrue:真多核多线程处理，类的函数统一返回sel

匹配file name:正则表达式匹配
"""
# =========================================================================
# import
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd
import time
import os
import shutil
import sys
sys.path.append(r'D:\myCodes\MVPA_LIChao\MVPA_Python\workstation')


# =========================================================================
# def
class copy_fmri():

    def __init__(
            sel,
            referencePath=r'E:\wangfeidata\folder.txt',
            regularExpressionOfsubjName_forReference='([1-9]\d*)',
            ith_reference=0,
            folderNameContainingFile_forSelect='',
            num_countBackwards=2,
            regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',
            ith_subjName=0,
            keywordThatFileContain='nii',
            neuroimageDataPath=r'E:\wangfeidata\FunImgARWD',
            savePath=r'E:\wangfeidata',
            n_processess=2,
            ifSaveLog=1,
            ifCopy=0,
            ifMove=0,
            saveInToOneOrMoreFolder='saveToEachSubjFolder',
            saveNameSuffix='.nii',
            ifRun=0):

        # =========================================================================
        sel.referencePath = referencePath
        sel.regularExpressionOfsubjName_forReference = regularExpressionOfsubjName_forReference
        sel.ith_reference = ith_reference
        sel.folderNameContainingFile_forSelect = folderNameContainingFile_forSelect
        sel.num_countBackwards = num_countBackwards
        sel.regularExpressionOfSubjName_forNeuroimageDataFiles = regularExpressionOfSubjName_forNeuroimageDataFiles
        sel.ith_subjName = ith_subjName
        sel.keywordThatFileContain = keywordThatFileContain
        sel.neuroimageDataPath = neuroimageDataPath
        sel.savePath = savePath
        sel.n_processess = n_processess
        sel.ifSaveLog = ifSaveLog
        sel.ifCopy = ifCopy
        sel.ifMove = ifMove
        sel.saveInToOneOrMoreFolder = saveInToOneOrMoreFolder
        sel.saveNameSuffix = saveNameSuffix
        sel.ifRun = ifRun

        # 核对参数信息
        if sel.ifCopy == 1 & sel.ifMove == 1:
            print('### Cannot copy and move at the same time! ###\n')
            print('### please press Ctrl+C to close the progress ###\n')

        # 新建结果保存文件夹
        if not os.path.exists(sel.savePath):
            os.makedirs(sel.savePath)

        # 读取referencePath(excel or text)
        try:
            sel.subjName_forSelect = pd.read_excel(
                sel.referencePath, dtype='str', header=None, index=None)
        except BaseException:
            sel.subjName_forSelect = pd.read_csv(
                sel.referencePath, dtype='str', header=None)
        #
        print('###提取subjName_forSelect中的匹配成分，默认为数字###\n###当有多个匹配时默认是第1个###\n')
#        ith_reference=sel.ith_reference
#        sel.ith_reference=0
        if sel.regularExpressionOfsubjName_forReference:
            sel.subjName_forSelect = sel.subjName_forSelect.iloc[:, 0]\
                .str.findall('[1-9]\d*')

            sel.subjName_forSelect = [sel.subjName_forSelect_[sel.ith_reference]
                                      for sel.subjName_forSelect_ in
                                      sel.subjName_forSelect
                                      if len(sel.subjName_forSelect_)]
# ===================================================================

    def walkAllPath(sel):
        sel.allWalkPath = os.walk(sel.neuroimageDataPath)
#        allWalkPath=[allWalkPath_ for allWalkPath_ in allWalkPath]
        return sel

    def fetch_allFilePath(sel):
        sel.allFilePath = []
        for onePath in sel.allWalkPath:
            for oneFile in onePath[2]:
                path = os.path.join(onePath[0], oneFile)
                sel.allFilePath.append(path)
        return sel

    def fetch_allSubjName(sel):
        '''
        num_countBackwards:subjName在倒数第几个block内(第一个计数为1)
        # 如'D:\myCodes\workstation_20180829_dynamicFC\FunImgARW\1-500\00002_resting\dti\dic.txt'
        # 的subjName在倒数第3个中
        '''
        sel.allSubjName = sel.allFilePath
        for i in range(sel.num_countBackwards - 1):
            sel.allSubjName = [os.path.dirname(
                allFilePath_) for allFilePath_ in sel.allSubjName]
        sel.allSubjName = [os.path.basename(
            allFilePath_) for allFilePath_ in sel.allSubjName]
        sel.allSubjName = pd.DataFrame(sel.allSubjName)
        sel.allSubjName_raw = sel.allSubjName
        return sel

    def fetch_folerNameContainingFile(sel):
        '''
        如果file上一级folder不是subject name，那么就涉及到选择那个文件夹下的file
        此时先确定每一个file上面的folder name(可能是模态名)，然后根据你的关键词来筛选
        '''
        sel.folerNameContainingFile = [os.path.dirname(
            allFilePath_) for allFilePath_ in sel.allFilePath]
        sel.folerNameContainingFile = [os.path.basename(
            folderName) for folderName in sel.folerNameContainingFile]
        return sel

    def fetch_allFileName(sel):
        '''
        获取把所有file name，用于后续的筛选。
        适用场景：假如跟file一起的有我们不需要的file，
        比如混杂在dicom file中的有text文件，而这些text是我们不想要的。
        '''
        sel.allFileName = [os.path.basename(
            allFilePath_) for allFilePath_ in sel.allFilePath]
        return sel
    # ===================================================================

    def screen_pathLogicalLocation_accordingTo_yourSubjName(sel):
        # 匹配subject name：注意此处用精确匹配，只有完成匹配时，才匹配成功
        # maker sure subjName_forSelect is pd.Series and its content is string

        if isinstance(sel.subjName_forSelect, type(pd.DataFrame([1]))):
            sel.subjName_forSelect = sel.subjName_forSelect.iloc[:, 0]
        if not isinstance(sel.subjName_forSelect[0], str):
            sel.subjName_forSelect = pd.Series(
                sel.subjName_forSelect, dtype='str')

        # 一定要注意匹配对之间的数据类型要一致！！！
        try:
            # 提取所有被试的folder
            #        sel.logic_index_subjname=\
            #                    np.sum(
            #                            pd.DataFrame(
            #                                    [sel.allSubjName.iloc[:,0].str.contains\
            #                                    (name_for_sel) for name_for_sel in sel.subjName_forSelect]
            #                                        ).T,
            #                            axis=1)
            #
            #        sel.logic_index_subjname=sel.logic_index_subjname>=1

            sel.allSubjName = sel.allSubjName.iloc[:, 0].str.findall(
                sel.regularExpressionOfSubjName_forNeuroimageDataFiles)

            # 正则表达提取后，可能有的不匹配而为空list,此时应该把空list当作不匹配而去除
            allSubjName_temp = []
#                sel.ith_subjName=1
            for name in sel.allSubjName.values:
                if name:
                    allSubjName_temp.append(name[sel.ith_subjName])
                else:
                    allSubjName_temp.append(None)
            sel.allSubjName = allSubjName_temp
            sel.allSubjName = pd.DataFrame(sel.allSubjName)
            sel.subjName_forSelect = pd.DataFrame(sel.subjName_forSelect)
            sel.logic_index_subjname = pd.DataFrame(
                np.zeros(len(sel.allSubjName)) == 1)
            for i in range(len(sel.subjName_forSelect)):
                sel.logic_index_subjname = sel.logic_index_subjname.mask(
                    sel.allSubjName == sel.subjName_forSelect.iloc[i, 0], True)

        except BaseException:
            print('subjName mismatch subjName_forSelected!\nplease check their type')
            sys.exit(0)

        return sel

    def screen_pathLogicalLocation_accordingTo_folerNameContainingFile(sel):
        # 匹配folerNameContainingFile：注意此处用的连续模糊匹配，只要含有这个关键词，则匹配
        if sel.folderNameContainingFile_forSelect:
            sel.logic_index_foler_name_containing_file = [
                sel.folderNameContainingFile_forSelect in oneName_ for oneName_ in sel.folerNameContainingFile]
            sel.logic_index_foler_name_containing_file = pd.DataFrame(
                sel.logic_index_foler_name_containing_file)
        else:
            sel.logic_index_foler_name_containing_file = np.ones(
                [len(sel.folerNameContainingFile), 1]) == 1
            sel.logic_index_foler_name_containing_file = pd.DataFrame(
                sel.logic_index_foler_name_containing_file)
        return sel

    def screen_pathLogicalLocation_accordingTo_fileName(sel):
        # 匹配file name:正则表达式匹配
        if sel.keywordThatFileContain:
            sel.allFileName = pd.Series(sel.allFileName)
            sel.logic_index_file_name = sel.allFileName.str.contains(
                sel.keywordThatFileContain)
        else:
            sel.logic_index_file_name = np.ones([len(sel.allFileName), 1]) == 1
            sel.logic_index_file_name = pd.DataFrame(sel.logic_index_file_name)

        return sel

    def fetch_totalLogicalLocation(sel):

        sel.logic_index_all = pd.concat(
            [
                sel.logic_index_file_name,
                sel.logic_index_foler_name_containing_file,
                sel.logic_index_subjname],
            axis=1)
        sel.logic_index_all = np.sum(
            sel.logic_index_all,
            axis=1) == np.shape(
            sel.logic_index_all)[1]
        return sel

    def fetch_selectedFilePath_accordingPathLogicalLocation(sel):
        # path
        sel.allFilePath = pd.DataFrame(sel.allFilePath)
        sel.allSelectedFilePath = sel.allFilePath[sel.logic_index_all]
        sel.allSelectedFilePath = sel.allSelectedFilePath.dropna()
        # folder name
        sel.allSubjName = pd.DataFrame(sel.allSubjName)
        sel.allSelectedSubjName = sel.allSubjName[sel.logic_index_all]
        sel.allSelectedSubjName = sel.allSelectedSubjName.dropna()
        # raw name
        sel.allSubjName_raw = pd.DataFrame(sel.allSubjName_raw)
        sel.allSelectedSubjName_raw = sel.allSubjName_raw[sel.logic_index_all]
        sel.allSelectedSubjName_raw = sel.allSelectedSubjName_raw.dropna()

        return sel
# ===================================================================

    def copy_allDicomsOfOneSubj(sel, i, subjName):
        n_allSelectedSubj = len(np.unique(sel.allSelectedSubjName_raw))
#        print('Copying the {}/{}th subject: {}...'.format(i+1,n_allSelectedSubj,subjName))

        # 每个file保存到每个subjxxx文件夹下面
        if sel.saveInToOneOrMoreFolder == 'saveToEachSubjFolder':
            output_folder = os.path.join(sel.savePath, subjName)
            # 新建subjxxx文件夹
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        # 所有file保存到一个folder下面（file的名字以subjxxx命名）
        elif sel.saveInToOneOrMoreFolder == 'saveToOneFolder':
            output_folder = os.path.join(
                sel.savePath, subjName + sel.saveNameSuffix)

        # copying OR moving OR do nothing
        fileIndex = sel.allSelectedSubjName_raw[(
            sel.allSelectedSubjName_raw.values == subjName)].index.tolist()
        if sel.ifCopy == 1 and sel.ifMove == 0:
            [shutil.copy(sel.allSelectedFilePath.loc[fileIndex_, :][0],
                         output_folder) for fileIndex_ in fileIndex]
        elif sel.ifCopy == 0 and sel.ifMove == 1:
            [shutil.move(sel.allSelectedFilePath.loc[fileIndex_, :][0],
                         output_folder) for fileIndex_ in fileIndex]
        elif sel.ifCopy == 0 and sel.ifMove == 0:
            print('### No copy and No move ###\n')
        else:
            print('### Cannot copy and move at the same time! ###\n')

        print('Copy the {}/{}th subject: {} OK!\n'.format(i + \
              1, n_allSelectedSubj, subjName))
    #

    def copy_allDicomsOfAllSubj_multiprocess(sel):

        s = time.time()

        # 每个file保存到每个subjxxx文件夹下面
        if sel.saveInToOneOrMoreFolder == 'saveToEachSubjFolder':
            pass
        elif sel.saveInToOneOrMoreFolder == 'saveToOneFolder':
            pass
        else:
            print(
                "###没有指定复制到一个文件夹还是每个被试文件夹###\n###{}跟'saveToOneFolder' OR 'saveToEachSubjFolder'都不符合###".format(
                    sel.saveInToOneOrMoreFolder))

        # 多线程
        # unique的name
        uniSubjName = sel.allSelectedSubjName_raw.iloc[:, 0].unique()

        print('Copying...\n')
        # 单线程
        #        for i,subjName in enumerate(uniSubjName):
        #            sel.copy_allDicomsOfOneSubj(i,subjName)

        # 多线程
        cores = multiprocessing.cpu_count()
        if sel.n_processess > cores:
            sel.n_processess = cores - 1

        with ThreadPoolExecutor(sel.n_processess) as executor:
            for i, subjName in enumerate(uniSubjName):
                task = executor.submit(
                    sel.copy_allDicomsOfOneSubj, i, subjName)
        #                print(task.done())

        print('=' * 30)
        #
        e = time.time()
        print('Done!\nRunning time is {:.1f} second'.format(e - s))
# ===================================================================

    def main_run(sel):
        # all path and name
        sel = sel.walkAllPath()
        sel = sel.fetch_allFilePath()
        sel = sel.fetch_allSubjName()
        sel = sel.fetch_allFileName()
        # select
        sel = sel.fetch_folerNameContainingFile()
        # logicLoc_subjName：根据被试名字匹配所得到的logicLoc。以此类推。
        # fileName≠subjName,比如fileName可以是xxx.nii,但是subjName可能是subjxxx
        sel = sel.screen_pathLogicalLocation_accordingTo_yourSubjName()
        sel = sel.screen_pathLogicalLocation_accordingTo_folerNameContainingFile()
        sel = sel.screen_pathLogicalLocation_accordingTo_fileName()
        sel = sel.fetch_totalLogicalLocation()
        sel = sel.fetch_selectedFilePath_accordingPathLogicalLocation()

        sel.unmatched_ref = \
            pd.DataFrame(list(
                set.difference(set(list(sel.subjName_forSelect.astype(np.int32).iloc[:, 0])),
                               set(list(sel.allSelectedSubjName.astype(np.int32).iloc[:, 0])))
            )
            )

        print('=' * 50 + '\n')
        print(
            'Files that not found are : {}\n\nThey may be saved in:\n[{}]\n'.format(
                sel.unmatched_ref.values,
                sel.savePath))
        print('=' * 50 + '\n')

        # save for checking
        if sel.ifSaveLog:
            now = time.localtime()
            now = time.strftime("%Y-%m-%d %H:%M:%S", now)

            #
            uniSubjName = sel.allSelectedSubjName.iloc[:, 0].unique()
            uniSubjName = [uniSubjName_ for uniSubjName_ in uniSubjName]
            uniSubjName = pd.DataFrame(uniSubjName)
            sel.allSelectedFilePath.to_csv(
                os.path.join(
                    sel.savePath,
                    'log_allSelectedFilePath.txt'),
                index=False,
                header=False)
            allSelectedSubjPath = [os.path.dirname(
                allSelectedFilePath_) for allSelectedFilePath_ in sel.allSelectedFilePath.iloc[:, 0]]
            allSelectedSubjPath = pd.DataFrame(
                allSelectedSubjPath).drop_duplicates()
            allSelectedSubjPath.to_csv(
                os.path.join(
                    sel.savePath,
                    'log_allSelectedSubjPath.txt'),
                index=False,
                header=False)
            uniSubjName.to_csv(
                os.path.join(
                    sel.savePath,
                    'log_allSelectedSubjName.txt'),
                index=False,
                header=False)

            sel.unmatched_ref.to_csv(
                os.path.join(
                    sel.savePath,
                    'log_unmatched_reference.txt'),
                index=False,
                header=False)
            pd.unique(
                sel.allSubjName).to_csv(
                os.path.join(
                    sel.savePath,
                    'log_allSubjName.txt'),
                index=False,
                header=False)
            #

            f = open(os.path.join(sel.savePath, "log_copy_inputs.txt"), 'a')
            f.write("\n\n")
            f.write('====================' + now + '====================')
            f.write("\n\n")
            f.write("referencePath is: " + sel.referencePath)
            f.write("\n\n")
            f.write(
                "folderNameContainingFile_forSelect are: " +
                sel.folderNameContainingFile_forSelect)
            f.write("\n\n")
            f.write("num_countBackwards is: " + str(sel.num_countBackwards))
            f.write("\n\n")
            f.write("regularExpressionOfSubjName_forNeuroimageDataFiles is: " +
                    str(sel.regularExpressionOfSubjName_forNeuroimageDataFiles))
            f.write("\n\n")
            f.write("keywordThatFileContain is: " +
                    str(sel.keywordThatFileContain))
            f.write("\n\n")
            f.write("neuroimageDataPath is: " + sel.neuroimageDataPath)
            f.write("\n\n")
            f.write("savePath is: " + sel.savePath)
            f.write("\n\n")
            f.write("n_processess is: " + str(sel.n_processess))
            f.write("\n\n")
            f.close()

        # copy
        if sel.ifRun:
            sel.copy_allDicomsOfAllSubj_multiprocess()
        return sel


if __name__ == '__main__':

    import copy_test as copy

    path = r'J:\Research_2017go\GAD\Data_Raw\Patients_WithSleepDisorder'
    folder = r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Utils\subj_id.xlsx'
    save_path = r'J:\Research_2017go\GAD\Data_Raw\test'

    sel = copy.copy_fmri(
        referencePath=folder,
        regularExpressionOfsubjName_forReference='([1-9]\d*)',
        ith_reference=0,
        folderNameContainingFile_forSelect='T1W',
        num_countBackwards=3,
        regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',
        ith_subjName=1,
        keywordThatFileContain='',
        neuroimageDataPath=path,
        savePath=save_path,
        n_processess=6,
        ifSaveLog=1,
        ifCopy=1,
        ifMove=0,
        saveInToOneOrMoreFolder='saveToEachSubjFolder',
        saveNameSuffix='',
        ifRun=1)

    result = sel.main_run()

#    results=result.__dict__
#    print(results.keys())
#    print('Done!')
