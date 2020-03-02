
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
    #3   folderNameContainingFile_forSelect:想把被试的哪个模态/或那个文件夹下的文件复制出来（如同时有'resting'和'dti'时，选择那个模态）
    #4   num_countBackwards:subjName在倒数第几个block内(第一个计数为1)
    #   如'D:\myCodes\workstation_20180829_dynamicFC\FunImgARW\1-500\00002_resting\dti\dic.txt'
    #  的subjName在倒数第3个中
    #5   regularExpressionOfSubjName_forNeuroimageDataFiles:用来筛选mri数据中subject name字符串的正则表达式
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
new featrue:真多核多线程处理，类的函数统一返回self
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
#from lc_selectFile_ import selectFile
#from sklearn.externals.joblib import Parallel, delayed
# =========================================================================
# def


class copy_fmri():

    def __init__(
            self,
            referencePath=r'E:\wangfeidata\folder.txt',
            regularExpressionOfsubjName_forReference='([1-9]\d*)',
            folderNameContainingFile_forSelect='',
            num_countBackwards=2,
            regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',
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
        self.referencePath = referencePath
        self.regularExpressionOfsubjName_forReference = regularExpressionOfsubjName_forReference
        self.folderNameContainingFile_forSelect = folderNameContainingFile_forSelect
        self.num_countBackwards = num_countBackwards
        self.regularExpressionOfSubjName_forNeuroimageDataFiles = regularExpressionOfSubjName_forNeuroimageDataFiles
        self.keywordThatFileContain = keywordThatFileContain
        self.neuroimageDataPath = neuroimageDataPath
        self.savePath = savePath
        self.n_processess = n_processess
        self.ifSaveLog = ifSaveLog
        self.ifCopy = ifCopy
        self.ifMove = ifMove
        self.saveInToOneOrMoreFolder = saveInToOneOrMoreFolder
        self.saveNameSuffix = saveNameSuffix
        self.ifRun = ifRun

        # 核对参数信息
        if self.ifCopy == 1 & self.ifMove == 1:
            print('### Cannot copy and move at the same time! ###\n')
            print('### please press Ctrl+C to close the progress ###\n')

        # 新建结果保存文件夹
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        # 读取referencePath(excel or text)
        try:
            self.subjName_forSelect = pd.read_excel(
                self.referencePath, dtype='str', header=None, index=None)
        except BaseException:
            self.subjName_forSelect = pd.read_csv(
                self.referencePath, dtype='str', header=None)
        #
        print('###提取subjName_forSelect中的匹配成分，默认为数字###\n###当有多个匹配时默认是第1个###\n')
        ith = 0
        if self.regularExpressionOfsubjName_forReference:
            self.subjName_forSelect = self.subjName_forSelect.iloc[:, 0]\
                .str.findall('[1-9]\d*')

            self.subjName_forSelect = [self.subjName_forSelect_[ith]
                                       for self.subjName_forSelect_ in
                                       self.subjName_forSelect
                                       if len(self.subjName_forSelect_)]
# ===================================================================

    def walkAllPath(self):
        self.allWalkPath = os.walk(self.neuroimageDataPath)
#        allWalkPath=[allWalkPath_ for allWalkPath_ in allWalkPath]
        return self

    def fetch_allFilePath(self):
        self.allFilePath = []
        for onePath in self.allWalkPath:
            for oneFile in onePath[2]:
                path = os.path.join(onePath[0], oneFile)
                self.allFilePath.append(path)
        return self

    def fetch_allSubjName(self):
        '''
        num_countBackwards:subjName在倒数第几个block内(第一个计数为1)
        # 如'D:\myCodes\workstation_20180829_dynamicFC\FunImgARW\1-500\00002_resting\dti\dic.txt'
        # 的subjName在倒数第3个中
        '''
        self.allSubjName = self.allFilePath
        for i in range(self.num_countBackwards - 1):
            self.allSubjName = [os.path.dirname(
                allFilePath_) for allFilePath_ in self.allSubjName]
        self.allSubjName = [os.path.basename(
            allFilePath_) for allFilePath_ in self.allSubjName]
        self.allSubjName = pd.DataFrame(self.allSubjName)
        self.allSubjName_raw = self.allSubjName
        return self

    def fetch_folerNameContainingFile(self):
        '''
        如果file上一级folder不是subject name，那么就涉及到选择那个文件夹下的file
        此时先确定每一个file上面的folder name(可能是模态名)，然后根据你的关键词来筛选
        '''
        self.folerNameContainingFile = [os.path.dirname(
            allFilePath_) for allFilePath_ in self.allFilePath]
        self.folerNameContainingFile = [os.path.basename(
            folderName) for folderName in self.folerNameContainingFile]
        return self

    def fetch_allFileName(self):
        '''
        获取把所有file name，用于后续的筛选。
        适用场景：假如跟file一起的有我们不需要的file，
        比如混杂在dicom file中的有text文件，而这些text是我们不想要的。
        '''
        self.allFileName = [os.path.basename(
            allFilePath_) for allFilePath_ in self.allFilePath]
        return self
    # ===================================================================

    def screen_pathLogicalLocation_accordingTo_yourSubjName(self):
        # 匹配subject name：注意此处用精确匹配，只有完成匹配时，才匹配成功
        # maker sure subjName_forSelect is pd.Series and its content is string

        if isinstance(self.subjName_forSelect, type(pd.DataFrame([1]))):
            self.subjName_forSelect = self.subjName_forSelect.iloc[:, 0]
        if not isinstance(self.subjName_forSelect[0], str):
            self.subjName_forSelect = pd.Series(
                self.subjName_forSelect, dtype='str')

        # 一定要注意匹配对之间的数据类型要一致！！！
        try:
            # 提取所有被试的folder
            self.allSubjName = self.allSubjName.iloc[:, 0].str.findall(
                self.regularExpressionOfSubjName_forNeuroimageDataFiles)

            # 正则表达提取后，可能有的不匹配而为空list,此时应该把空list当作不匹配而去除
            allSubjName_temp = []
            for name in self.allSubjName.values:
                if name:
                    allSubjName_temp.append(name[0])
                else:
                    allSubjName_temp.append(None)
            self.allSubjName = allSubjName_temp
            self.allSubjName = pd.DataFrame(self.allSubjName)
            self.subjName_forSelect = pd.DataFrame(self.subjName_forSelect)
            self.logic_index_subjname = pd.DataFrame(
                np.zeros(len(self.allSubjName)) == 1)
            for i in range(len(self.subjName_forSelect)):
                self.logic_index_subjname = self.logic_index_subjname.mask(
                    self.allSubjName == self.subjName_forSelect.iloc[i, 0], True)

        except BaseException:
            print('subjName mismatch subjName_forSelected!\nplease check their type')
            sys.exit(0)

        return self

    def screen_pathLogicalLocation_accordingTo_folerNameContainingFile(self):
        # 匹配folerNameContainingFile：注意此处用的连续模糊匹配，只要含有这个关键词，则匹配
        if self.folderNameContainingFile_forSelect:
            self.logic_index_foler_name_containing_file = [
                self.folderNameContainingFile_forSelect in oneName_ for oneName_ in self.folerNameContainingFile]
            self.logic_index_foler_name_containing_file = pd.DataFrame(
                self.logic_index_foler_name_containing_file)
        else:
            self.logic_index_foler_name_containing_file = np.ones(
                [len(self.folerNameContainingFile), 1]) == 1
            self.logic_index_foler_name_containing_file = pd.DataFrame(
                self.logic_index_foler_name_containing_file)
        return self

    def screen_pathLogicalLocation_accordingTo_fileName(self):
        # 匹配file name:正则表达式匹配
        if self.keywordThatFileContain:
            self.allFileName = pd.Series(self.allFileName)
            self.logic_index_file_name = self.allFileName.str.contains(
                self.keywordThatFileContain)
        else:
            self.logic_index_file_name = np.ones(
                [len(self.allFileName), 1]) == 1
            self.logic_index_file_name = pd.DataFrame(
                self.logic_index_file_name)

        return self

    def fetch_totalLogicalLocation(self):

        self.logic_index_all = pd.concat(
            [
                self.logic_index_file_name,
                self.logic_index_foler_name_containing_file,
                self.logic_index_subjname],
            axis=1)
        self.logic_index_all = np.sum(
            self.logic_index_all,
            axis=1) == np.shape(
            self.logic_index_all)[1]
        return self

    def fetch_selectedFilePath_accordingPathLogicalLocation(self):
        # path
        self.allFilePath = pd.DataFrame(self.allFilePath)
        self.allSelectedFilePath = self.allFilePath[self.logic_index_all]
        self.allSelectedFilePath = self.allSelectedFilePath.dropna()
        # folder name
        self.allSubjName = pd.DataFrame(self.allSubjName)
        self.allSelectedSubjName = self.allSubjName[self.logic_index_all]
        self.allSelectedSubjName = self.allSelectedSubjName.dropna()
        # raw name
        self.allSubjName_raw = pd.DataFrame(self.allSubjName_raw)
        self.allSelectedSubjName_raw = self.allSubjName_raw[self.logic_index_all]
        self.allSelectedSubjName_raw = self.allSelectedSubjName_raw.dropna()

        return self
# ===================================================================

    def copy_allDicomsOfOneSubj(self, i, subjName):
        n_allSelectedSubj = len(self.allSelectedSubjName_raw)
#        print('Copying the {}/{}th subject: {}...'.format(i+1,n_allSelectedSubj,subjName))

        # 每个file保存到每个subjxxx文件夹下面
        if self.saveInToOneOrMoreFolder == 'saveToEachSubjFolder':
            output_folder = os.path.join(self.savePath, subjName)
            # 新建subjxxx文件夹
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        # 所有file保存到一个folder下面（file的名字以subjxxx命名）
        elif self.saveInToOneOrMoreFolder == 'saveToOneFolder':
            output_folder = os.path.join(
                self.savePath, subjName + self.saveNameSuffix)

        # copying OR moving OR do nothing
        fileIndex = self.allSelectedSubjName_raw[(
            self.allSelectedSubjName_raw.values == subjName)].index.tolist()
        if self.ifCopy == 1 and self.ifMove == 0:
            [shutil.copy(self.allSelectedFilePath.loc[fileIndex_, :][0],
                         output_folder) for fileIndex_ in fileIndex]
        elif self.ifCopy == 0 and self.ifMove == 1:
            [shutil.move(self.allSelectedFilePath.loc[fileIndex_, :][0],
                         output_folder) for fileIndex_ in fileIndex]
        elif self.ifCopy == 0 and self.ifMove == 0:
            print('### No copy and No move ###\n')
        else:
            print('### Cannot copy and move at the same time! ###\n')

        print('Copy the {}/{}th subject: {} OK!\n'.format(i +
                                                          1, n_allSelectedSubj, subjName))
    #

    def copy_allDicomsOfAllSubj_multiprocess(self):

        s = time.time()

        # 每个file保存到每个subjxxx文件夹下面
        if self.saveInToOneOrMoreFolder == 'saveToEachSubjFolder':
            pass
        elif self.saveInToOneOrMoreFolder == 'saveToOneFolder':
            pass
        else:
            print(
                "###没有指定复制到一个文件夹还是每个被试文件夹###\n###{}跟'saveToOneFolder' OR 'saveToEachSubjFolder'都不符合###".format(
                    self.saveInToOneOrMoreFolder))
#            return -1

        # 多线程
        # unique的name
        uniSubjName = self.allSelectedSubjName_raw.iloc[:, 0].unique()

        print('Copying...\n')
        # 单线程
#        for i,subjName in enumerate(uniSubjName):
#            self.copy_allDicomsOfOneSubj(i,subjName)

        # 多线程
        cores = multiprocessing.cpu_count()
        if self.n_processess > cores:
            self.n_processess = cores - 1

        with ThreadPoolExecutor(self.n_processess) as executor:
            for i, subjName in enumerate(uniSubjName):
                task = executor.submit(
                    self.copy_allDicomsOfOneSubj, i, subjName)
#                print(task.done())

        print('=' * 30)
#
        e = time.time()
#        print('Done!\nRunning time is {:.1f} second'.format(e-s))
# ===================================================================

    def main_run(self):
        # all path and name
        self = self.walkAllPath()
        self = self.fetch_allFilePath()
        self = self.fetch_allSubjName()
        self = self.fetch_allFileName()
        # select
        self = self.fetch_folerNameContainingFile()
        # logicLoc_subjName：根据被试名字匹配所得到的logicLoc。以此类推。
        # fileName≠subjName,比如fileName可以是xxx.nii,但是subjName可能是subjxxx
        self = self.screen_pathLogicalLocation_accordingTo_yourSubjName()
        self = self.screen_pathLogicalLocation_accordingTo_folerNameContainingFile()
        self = self.screen_pathLogicalLocation_accordingTo_fileName()
        self = self.fetch_totalLogicalLocation()
        self = self.fetch_selectedFilePath_accordingPathLogicalLocation()

        self.unmatched_ref = \
            pd.DataFrame(list(
                set.difference(set(list(self.subjName_forSelect.astype(np.int32).iloc[:, 0])),
                               set(list(self.allSelectedSubjName.astype(np.int32).iloc[:, 0])))
            )
            )

        print('=' * 50 + '\n')
        print(
            'Files that not found are : {}\n\nThey may be saved in:\n[{}]\n'.format(
                self.unmatched_ref.values,
                self.savePath))
        print('=' * 50 + '\n')

        # save for checking
        if self.ifSaveLog:
            now = time.localtime()
            now = time.strftime("%Y-%m-%d %H:%M:%S", now)

            #
            uniSubjName = self.allSelectedSubjName.iloc[:, 0].unique()
            uniSubjName = [uniSubjName_ for uniSubjName_ in uniSubjName]
            uniSubjName = pd.DataFrame(uniSubjName)
            self.allSelectedFilePath.to_csv(
                os.path.join(
                    self.savePath,
                    'log_allSelectedFilePath.txt'),
                index=False,
                header=False)
            allSelectedSubjPath = [os.path.dirname(
                allSelectedFilePath_) for allSelectedFilePath_ in self.allSelectedFilePath.iloc[:, 0]]
            allSelectedSubjPath = pd.DataFrame(
                allSelectedSubjPath).drop_duplicates()
            allSelectedSubjPath.to_csv(
                os.path.join(
                    self.savePath,
                    'log_allSelectedSubjPath.txt'),
                index=False,
                header=False)
            uniSubjName.to_csv(
                os.path.join(
                    self.savePath,
                    'log_allSelectedSubjName.txt'),
                index=False,
                header=False)

            self.unmatched_ref.to_csv(
                os.path.join(
                    self.savePath,
                    'log_unmatched_reference.txt'),
                index=False,
                header=False)
            self.allSubjName.to_csv(
                os.path.join(
                    self.savePath,
                    'log_allSubjName.txt'),
                index=False,
                header=False)
            #

            f = open(os.path.join(self.savePath, "log_copy_inputs.txt"), 'a')
            f.write("\n\n")
            f.write('====================' + now + '====================')
            f.write("\n\n")
            f.write("referencePath is: " + self.referencePath)
            f.write("\n\n")
            f.write(
                "folderNameContainingFile_forSelect are: " +
                self.folderNameContainingFile_forSelect)
            f.write("\n\n")
            f.write("num_countBackwards is: " + str(self.num_countBackwards))
            f.write("\n\n")
            f.write("regularExpressionOfSubjName_forNeuroimageDataFiles is: " +
                    str(self.regularExpressionOfSubjName_forNeuroimageDataFiles))
            f.write("\n\n")
            f.write("keywordThatFileContain is: " +
                    str(self.keywordThatFileContain))
            f.write("\n\n")
            f.write("neuroimageDataPath is: " + self.neuroimageDataPath)
            f.write("\n\n")
            f.write("savePath is: " + self.savePath)
            f.write("\n\n")
            f.write("n_processess is: " + str(self.n_processess))
            f.write("\n\n")
            f.close()

        # copy
        if self.ifRun:
            self.copy_allDicomsOfAllSubj_multiprocess()
        return self


if __name__ == '__main__':

    import lc_copy_selected_file_V5 as copy

    path = r'J:\dynamicALFF\Results\static_ALFF\ALFF_FunImgARWDFCB'
    folder = r'J:\dynamicFC\state\folder_HC.xlsx'
    save_path = r'J:\dynamicALFF\Results\static_ALFF\test\HC_ALFF'

    sel = copy.copy_fmri(
        referencePath=folder,
        regularExpressionOfsubjName_forReference='([1-9]\d*)',
        folderNameContainingFile_forSelect='',
        num_countBackwards=1,
        regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',
        keywordThatFileContain='^ALFF',
        neuroimageDataPath=path,
        savePath=save_path,
        n_processess=6,
        ifSaveLog=0,
        ifCopy=1,
        ifMove=0,
        saveInToOneOrMoreFolder='saveToOneFolder',
        saveNameSuffix='',
        ifRun=1)

    result = sel.main_run()

#    results=result.__dict__
#    print(results.keys())
#    print('Done!')
