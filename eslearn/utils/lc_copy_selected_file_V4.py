# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:05:28 2018:
在版本3的基础上,根据pandas的join方法来求交集

根据从量表中筛选的样本，来获得符合要求的原始数据的路径
数据结构neuroimageDataPath//subject00001//files
也可以是任何的数据结构，只要给定subjName在哪里就行
总之，最后把file复制到其他地方（可以限定某个file）

input:
    # 1   referencePath:需要复制的被试名字所在text文件（大表中的folder）
    # 2   regularExpressionOfsubjName_forReference:如提取量表中subjName的正则表达式
    # 3   folderNameContainingFile_forSelect:想把被试的哪个模态/或那个文件夹下的文件复制出来（如同时有'resting'和'dti'时，选择那个模态）
    # 4   num_countBackwards:subjName在倒数第几个block内(第一个计数为1)
    #   如'D:\myCodes\workstation_20180829_dynamicFC\FunImgARW\1-500\00002_resting\dti\dic.txt'
    #  的subjName在倒数第3个中
    # 5   regularExpressionOfSubjName_forNeuroimageDataFiles:用来筛选mri数据中subject name字符串的正则表达式
    # 6   keywordThatFileContain:用来筛选file的正则表达式或keyword
    # 7   neuroimageDataPath：原始数据的根目录
    # 8   savePath: 将原始数据copy到哪个大路径
    #    n_processess=5几个线程
    # 9  ifSaveLog：是否保存复制log
    # 10  ifCopy：是否执行复制功能
    # 11 ifMove:是否移动（0）
    # 12  saveInToOneOrMoreFolder：保存到每个被试文件夹下，还是保存到一个文件夹下
    # 13  saveNameSuffix：文件保存的尾缀（'.nii'）
    # 14  ifRun:是否真正对文件执行移动或复制（0）
    #   总体来说被复制的文件放在如下的路径：savePath/saveFolderName/subjName/files
@author: LI Chao
"""
# =========================================================================
# import
import sys
import shutil
import os
import time
# from lc_selectFile_ import selectFile
import pandas as pd
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
# =========================================================================
# def


class copy_fmri():

    def __init__(self,
                 referencePath=r'E:\wangfeidata\folder.txt',
                 regularExpressionOfsubjName_forReference='([1-9]\d*)',
                 folderNameContainingFile_forSelect='',
                 num_countBackwards=2,
                 regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',
                 keywordThatFileContain='nii',
                 neuroimageDataPath=r'E:\wangfeidata\FunImgARWD',
                 savePath=r'E:\wangfeidata',
                 n_processess=5,
                 ifSaveLog=1,
                 ifCopy=0,
                 ifMove=0,
                 saveInToOneOrMoreFolder='saveToEachSubjFolder',
                 saveNameSuffix='.nii',
                 ifRun=0):

# 核对参数信息
        if ifCopy == 1 & ifMove == 1:
            print('### Cannot copy and move at the same time! ###\n')
            print('### please press Ctrl+C to close the progress ###\n')
            time.sleep(5)
#        print('==========================================================')
#        print('\nThe num_countBackwards that to screen subject name is {} !'.format(num_countBackwards))
#        print('\nKeyword of folder name that containing the files is {} !'.format(folderNameContainingFile_forSelect))
#        print('regularExpressionOfSubjName_forNeuroimageDataFiles is {}'.format(regularExpressionOfSubjName_forNeuroimageDataFiles))
#        print('ifCopy is {}'.format(ifCopy))
#        print('saveInToOneOrMoreFolder is {}'.format(saveInToOneOrMoreFolder))
#        print('==========================================================')
#        input("***请核对以上信息是否准确，否则复制出错!***")
# =========================================================================
        # accept excel or csv
        self.referencePath = referencePath
        try:
            self.subjName_forSelect = pd.read_excel(
    referencePath, dtype='str', header=None, index=None)
        except:
            self.subjName_forSelect = pd.read_csv(
                referencePath, dtype='str', header=None)
        #
        print('###提取subjName_forSelect中的匹配成分，默认为数字###\n###当有多个匹配时默认是第1个###\n')
        ith = 0
        if regularExpressionOfsubjName_forReference:
            self.subjName_forSelect = self.subjName_forSelect.iloc[:, 0]\
                                                .str.findall('[1-9]\d*')

            self.subjName_forSelect = [self.subjName_forSelect_[ith]
                                     for self.subjName_forSelect_ in
                                     self.subjName_forSelect
                                     if len(self.subjName_forSelect_)]
        # 提取subjName_forSelect完毕

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
# ===================================================================

    def walkAllPath(self):
        allWalkPath = os.walk(self.neuroimageDataPath)
#        allWalkPath=[allWalkPath_ for allWalkPath_ in allWalkPath]
        return allWalkPath

    def fetch_allFilePath(self, allWalkPath):
        allFilePath = []
        for onePath in allWalkPath:
                for oneFile in onePath[2]:
                    path = os.path.join(onePath[0], oneFile)
                    allFilePath.append(path)
        return allFilePath

    def fetch_allSubjName(self, allFilePath):
        '''
        num_countBackwards:subjName在倒数第几个block内(第一个计数为1)
        # 如'D:\myCodes\workstation_20180829_dynamicFC\FunImgARW\1-500\00002_resting\dti\dic.txt'
        # 的subjName在倒数第3个中
        '''
#        allWalkPath=sel.walkAllPath()
#        allFilePath=sel.fetch_allFilePath(allWalkPath)
        allSubjName = allFilePath
        for i in range(self.num_countBackwards - 1):
            allSubjName = [os.path.dirname(allFilePath_)
                                           for allFilePath_ in allSubjName]
        allSubjName = [os.path.basename(allFilePath_)
                                        for allFilePath_ in allSubjName]
        allSubjName = pd.DataFrame(allSubjName)
#        allSubjName=allSubjName.iloc[:,0].where(allSubjName.iloc[:,0]!='').dropna()
#        allSubjName=pd.DataFrame(allSubjName)
        return allSubjName

    def fetch_folerNameContainingFile(self, allFilePath):
        '''
        如果file上一级folder不是subject name，那么就涉及到选择那个文件夹下的file
        此时先确定每一个file上面的folder name(可能是模态名)，然后根据你的关键词来筛选
        '''
        folerNameContainingFile = [os.path.dirname(
            allFilePath_) for allFilePath_ in allFilePath]
        folerNameContainingFile = [os.path.basename(
            folderName) for folderName in folerNameContainingFile]
        return folerNameContainingFile

    def fetch_allFileName(self, allFilePath):
        '''
        获取把所有file name，用于后续的筛选。
        适用场景：假如跟file一起的有我们不需要的file，
        比如混杂在dicom file中的有text文件，而这些text是我们不想要的。
        '''
        allFileName = [os.path.basename(allFilePath_)
                                        for allFilePath_ in allFilePath]
        return allFileName
    # ===================================================================

    def screen_pathLogicalLocation_accordingTo_yourSubjName(self, allSubjName):
        # 匹配subject name：注意此处用精确匹配，只有完成匹配时，才匹配成功
        # maker sure subjName_forSelect is pd.Series and its content is string
        if type(self.subjName_forSelect) is type(pd.DataFrame([1])):
            self.subjName_forSelect = self.subjName_forSelect.iloc[:, 0]
        if type(self.subjName_forSelect[0]) is not str:
            self.subjName_forSelect = pd.Series(
                self.subjName_forSelect, dtype='str')

        # 一定要注意匹配对之间的数据类型要一致！！！
#        allSubjName=sel.fetch_allSubjName(allFilePath)
        try:
            allSubjName = allSubjName.iloc[:, 0].str.findall(
                self.regularExpressionOfSubjName_forNeuroimageDataFiles)
            # 正则表达后，可能有的不匹配而为空list,此时应该把空list当作不匹配而去除
            allSubjName_temp = []
            for name in allSubjName.values:
                if name:
                    allSubjName_temp.append(name[0])
                else:
                    allSubjName_temp.append(None)
            allSubjName = allSubjName_temp
            allSubjName = pd.DataFrame(allSubjName)
            self.subjName_forSelect = pd.DataFrame(self.subjName_forSelect)
#            self.subjName_forSelect
            intersect = allSubjName.set_index(0).join(
    self.subjName_forSelect.set_index(0), how='right')
            intersect = pd.Series(intersect.index)
            # allSubjName有，但是subjName_forSelect没有
#            self.difName=allSubjName.join(self.subjName_forSelect)
#            self.difName=self.difName.where(self.difName!='').dropna()
        except:
            print('subjName mismatch subjName_forSelected!\nplease check their type')
            sys.exit(0)
        if any(intersect):
            # 为了逻辑比较，将allSubjName 转化为DataFrame
            allSubjName = pd.DataFrame(allSubjName)
            logic_loc = [allSubjName == intersect_ for intersect_ in intersect]
            if len(logic_loc) > 1:
                logic_loc = pd.concat(logic_loc, axis=1)
                logic_loc = np.sum(logic_loc, axis=1)
                logic_loc = logic_loc == 1
            else:
                logic_loc = logic_loc
            logic_loc = pd.DataFrame(logic_loc)
        else:
            logic_loc = np.zeros([len(allSubjName), 1]) == 1
            logic_loc = pd.DataFrame(logic_loc)
        return logic_loc

    def screen_pathLogicalLocation_accordingTo_folerNameContainingFile(
        self, folerNameContainingFile):
        # 匹配folerNameContainingFile：注意此处用的连续模糊匹配，只要含有这个关键词，则匹配
        if self.folderNameContainingFile_forSelect:
            logic_loc = [
    self.folderNameContainingFile_forSelect in oneName_ for oneName_ in folerNameContainingFile]
            logic_loc = pd.DataFrame(logic_loc)
        else:
            logic_loc = np.ones([len(folerNameContainingFile), 1]) == 1
            logic_loc = pd.DataFrame(logic_loc)
        return logic_loc

    def screen_pathLogicalLocation_accordingTo_fileName(self, allFileName):
        # 匹配file name:注意此处用的连续模糊匹配，只要含有这个关键词，则匹配
        if self.keywordThatFileContain:
            logic_loc = [
    self.keywordThatFileContain in oneName_ for oneName_ in allFileName]
            logic_loc = pd.DataFrame(logic_loc)
        else:
            logic_loc = np.ones([len(allFileName), 1]) == 1
            logic_loc = pd.DataFrame(logic_loc)

        return logic_loc

    def fetch_totalLogicalLocation(self,
        logicLoc_subjName, logicLoc_folderNameContaningFile, logicLoc_fileName):

        logic_loc = pd.concat([logicLoc_subjName,
    logicLoc_folderNameContaningFile,
    logicLoc_fileName],
     axis=1)
        logic_loc = np.sum(logic_loc, axis=1) == np.shape(logic_loc)[1]
        return logic_loc

    def fetch_selectedFilePath_accordingPathLogicalLocation(self,
                                            allFilePath, allSubjName, logic_loc):
        #
        allFilePath = pd.DataFrame(allFilePath)
        allSelectedFilePath = allFilePath[logic_loc]
        allSelectedFilePath = allSelectedFilePath.dropna()
        # name
        allSubjName = pd.DataFrame(allSubjName)
        allSelectedSubjName = allSubjName[logic_loc]
        allSelectedSubjName = allSelectedSubjName.dropna()
        return allSelectedFilePath, allSelectedSubjName
# ===================================================================

    def copy_allDicomsOfOneSubj(
    self,
    i,
    subjName,
    allSelectedSubjName,
     allSelectedFilePath):
        n_allSelectedSubj = len(allSelectedSubjName)
        print('Copying the {}/{}th subject: {}...'.format(i +
              1, n_allSelectedSubj, subjName))

        # 每个file保存到每个subjxxx文件夹下面
        if self.saveInToOneOrMoreFolder == 'saveToEachSubjFolder':
            output_folder = os.path.join(self.savePath, subjName)
            # 新建subjxxx文件夹
            if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
        # 所有file保存到一个folder下面（file的名字以subjxxx命名）
        if self.saveInToOneOrMoreFolder == 'saveToOneFolder':
            output_folder = os.path.join(self.savePath,
                                       subjName + self.saveNameSuffix)

        # copying OR moving OR do nothing
        fileIndex = allSelectedSubjName[(
            allSelectedSubjName.values == subjName)].index.tolist()
        if self.ifCopy == 1 and self.ifMove == 0:
            [shutil.copy(allSelectedFilePath.loc[fileIndex_, :][0],
                         output_folder) for fileIndex_ in fileIndex]
        elif self.ifCopy == 0 and self.ifMove == 1:
            [shutil.move(allSelectedFilePath.loc[fileIndex_, :][0],
                         output_folder) for fileIndex_ in fileIndex]
        elif self.ifCopy == 0 and self.ifMove == 0:
            print('### No copy and No move ###\n')
        else:
            print('### Cannot copy and move at the same time! ###\n')

        print('OK!\n')
    #

    def copy_allDicomsOfAllSubj_multiprocess(self, allSelectedSubjName,
                                allSelectedFilePath):
        # 新建保存文件夹
        if not os.path.exists(self.savePath):
                os.makedirs(self.savePath)

        # 多线程
        s = time.time()
        # unique的name
        uniSubjName = allSelectedSubjName.iloc[:, 0].unique()
        # 当复制的文件较少时，不要开多线程
        if len(uniSubjName) <= 500:
            self.n_processess = 1

        print('Copying...\n')
        Parallel(n_jobs=self.n_processess, backend='threading')(delayed(self.copy_allDicomsOfOneSubj)(i, subjName, allSelectedSubjName, allSelectedFilePath)
             for i, subjName in enumerate(uniSubjName))
        e = time.time()
        print('Done!\nRunning time is {:.1f}'.format(e - s))
# ===================================================================

    def main_run(self):
        # all path and name
        allWalkPath = self.walkAllPath()
        allFilePath = self.fetch_allFilePath(allWalkPath)
        allSubjName = self.fetch_allSubjName(allFilePath)
        allFileName = self.fetch_allFileName(allFilePath)
        # select
        folderNameContainingFile = self.fetch_folerNameContainingFile(
            allFilePath)
        # logicLoc_subjName：根据被试名字匹配所得到的logicLoc。以此类推。
        # fileName≠subjName,比如fileName可以是xxx.nii,但是subjName可能是subjxxx
        logicLoc_subjName = self.screen_pathLogicalLocation_accordingTo_yourSubjName(
            allSubjName)
        logicLoc_folderNameContaningFile = self.screen_pathLogicalLocation_accordingTo_folerNameContainingFile(
            folderNameContainingFile)
        logicLoc_fileName = self.screen_pathLogicalLocation_accordingTo_fileName(
            allFileName)
        logic_loc = self.fetch_totalLogicalLocation(
    logicLoc_subjName, logicLoc_folderNameContaningFile, logicLoc_fileName)
        allSelectedFilePath, allSelectedSubjName = self.fetch_selectedFilePath_accordingPathLogicalLocation(
            allFilePath, allSubjName, logic_loc)

        # save for checking
        if self.ifSaveLog:
            now = time.localtime()
            now = time.strftime("%Y-%m-%d %H:%M:%S", now)
            #
            uniSubjName = allSelectedSubjName.iloc[:, 0].unique()
            uniSubjName = [uniSubjName_ for uniSubjName_ in uniSubjName]
            uniSubjName = pd.DataFrame(uniSubjName)
            allSelectedFilePath.to_csv(
    os.path.join(
        self.savePath,
        'log_allSelectedFilePath.txt'),
        index=False,
         header=False)
            allSelectedSubjPath = [os.path.dirname(
                allSelectedFilePath_) for allSelectedFilePath_ in allSelectedFilePath.iloc[:, 0]]
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
            self.difName.to_csv(
    os.path.join(
        self.savePath,
        'log_difdSubjName.txt'),
        index=False,
         header=False)
            allSubjName.to_csv(
    os.path.join(
        self.savePath,
        'log_allSubjName.txt'),
        index=False,
         header=False)
            #
            if len(uniSubjName) <= 100:
                self.n_processess = 1
            f = open(os.path.join(self.savePath, "copy_inputs.txt"), 'a')
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
            self.copy_allDicomsOfAllSubj_multiprocess(
                allSelectedSubjName, allSelectedFilePath)
        return allFilePath, allSubjName, logic_loc, allSelectedFilePath, allSelectedSubjName


if __name__ == '__main__':
     import lc_copy_selected_file_V4 as copy
    # basic['folder'].to_csv(r'I:\dynamicALFF\folder.txt',header=False,index=False)
    path=r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018_07_DynamicFC_insomnia\FunImgARWS'                    
    folder=r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018_07_DynamicFC_insomnia\folder.txt'
    sel=copy.copy_fmri(referencePath=folder,
                      regularExpressionOfsubjName_forReference='([1-9]\d*)',
                      folderNameContainingFile_forSelect='',
                      num_countBackwards=2,
                      regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',\
                      keywordThatFileContain='nii',
                      neuroimageDataPath=path,
                      savePath=r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018_07_DynamicFC_insomnia\test',
                      n_processess=5,
                      ifSaveLog=1,
                      ifCopy=1,
                      ifMove=0,
                      saveInToOneOrMoreFolder='saveToEachSubjFolder',
                      saveNameSuffix='',
                      ifRun=0)
    
    allFilePath,allSubjName,\
    logic_loc,allSelectedFilePath,allSelectedSubjName=\
    sel.main_run()
    print('Done!')
