# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:05:28 2018:
在版本3的基础上,根据pandas的join方法来求交集

根据从量表中筛选的样本，来获得符合要求的原始数据的路径
数据结构neuroimageDataPath//subject00001//files
也可以是任何的数据结构，只要给定subjName在哪里就行
总之，最后把file复制到其他地方（可以给每个subject限定某个符合条件file，比如以'.nii'结尾的file）

input:
    #   reference_file:需要复制的被试名字所在text文件（大表中的uid）
    #   keywork_of_reference_uid:如提取量表中唯一识别号的正则表达式
    #   ith_number_of_reference_uid: 量表中的唯一识别号有多个匹配项时，选择第几个 （比如有一个名字为subj0001_bold7000, 此时可能匹配到0001和7000，遇到这种情况选择第几个匹配项）
    #   keywork_of_target_file: 需要移动/赋值的文件的关键字（除uid之外的）
    #   keyword_of_parent_folder_containing_target_file:想把被试的哪个模态/或那个文件夹下的文件复制出来（如同时有'resting'和'dti'时，选择那个模态）
    #   unique_id_level_of_target_file:与referenceid匹配的唯一识别号在倒数第几个block内(以target file为起点计算，第一个计数为1)
    #   如'D:\myCodes\workstation_20180829_dynamicFC\FunImgARW\1-500\00002_resting\dti\dic.txt'的唯一识别号在倒数第3个中
    #   keyword_of_target_file_uid:用来筛选mri数据中唯一识别号的正则表达式
    #   ith_number_of_targetfile_uid: target file中的唯一识别号有多个匹配项时，选择第几个.
    #   keyword_of_target_file_uid:用来筛选file的正则表达式或keyword
    #   targe_file_folder：原始数据的根目录
    #   out_path: 将原始数据copy到哪个大路径
    #   n_processess=5几个线程
    #   is_save_log：是否保存复制log
    #  is_copy：是否执行复制功能
    #  is_move:是否移动（0）
    #  save_into_one_or_more_folder：'one_file_one_folder' or 'all_files_in_one_folder'
    #  save_suffix：文件保存的尾缀（'.nii'）
    #  is_run:是否真正对文件执行移动或复制（0）
    #   总体来说被复制的文件放在如下的路径：out_path/saveFolderName/subjName/files
@author: LI Chao
new featrue:真多核多线程处理，类的函数统一返回self
匹配file name:正则表达式匹配
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import time
import os
import shutil
import sys
sys.path.append(
    r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')


class CopyFmri():
    def __init__(
            self,
            reference_file=None,
            targe_file_folder=None,

            keywork_of_reference_uid='([1-9]\d*)',
            ith_number_of_reference_uid=0,
            keyword_of_target_file_uid='([1-9]\d*)',
            ith_number_of_targetfile_uid=0,
            unique_id_level_of_target_file=2,
            
            keywork_of_target_file='nii',
            keyword_of_parent_folder_containing_target_file='',

            out_path=None,
            n_processess=2,
            is_save_log=1,
            is_copy=0,
            is_move=0,
            save_into_one_or_more_folder='one_file_one_folder',
            save_suffix='',
            is_run=0):

        self.reference_file = reference_file
        self.targe_file_folder = targe_file_folder

        self.keywork_of_reference_uid = keywork_of_reference_uid
        self.ith_number_of_reference_uid = ith_number_of_reference_uid
        self.keyword_of_target_file_uid = keyword_of_target_file_uid
        self.unique_id_level_of_target_file = unique_id_level_of_target_file
        self.ith_number_of_targetfile_uid = ith_number_of_targetfile_uid

        self.keywork_of_target_file = keywork_of_target_file
        self.keyword_of_parent_folder_containing_target_file = keyword_of_parent_folder_containing_target_file

        self.out_path = out_path
        self.n_processess = n_processess
        self.is_save_log = is_save_log
        self.is_copy = is_copy
        self.is_move = is_move
        self.save_into_one_or_more_folder = save_into_one_or_more_folder
        self.save_suffix = save_suffix
        self.is_run = is_run

    # %% process the input
    def _after_init(self):
        """ handle the init parameter
        """

        # chech param
        if self.is_copy == 1 & self.is_move == 1:
            raise ValueError('Cannot copy and move at the same time!')
            print('### please press Ctrl+C to close the progress ###\n')

        # create save folder
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        # read reference_file(excel or text)
        try:
            self.subjName_forSelect = pd.read_excel(
                self.reference_file, dtype='str', header=None, index=None)
        except BaseException:
            self.subjName_forSelect = pd.read_csv(
                self.reference_file, dtype='str', header=None)
        print('###提取subjName_forSelect中的匹配成分，默认为数字###\n###当有多个匹配时默认是第1个###\n')
        if self.keywork_of_reference_uid:
            self.subjName_forSelect = self.subjName_forSelect.iloc[:, 0].str.findall(self.keywork_of_reference_uid)

            self.subjName_forSelect = [self.subjName_forSelect_[self.ith_number_of_reference_uid]
                                      for self.subjName_forSelect_ in
                                      self.subjName_forSelect
                                      if len(self.subjName_forSelect_)]

    def walkAllPath(self):
        self.allWalkPath = os.walk(self.targe_file_folder)
#        allWalkPath=[allWalkPath_ for allWalkPath_ in allWalkPath]
        return self

    def fetch_allFilePath(self):
        self.allFilePath = []
        for onePath in self.allWalkPath:
            for oneFile in onePath[2]:
                source_folder = os.path.join(onePath[0], oneFile)
                self.allFilePath.append(source_folder)
        return self

    def fetch_allSubjName(self):
        '''
        unique_id_level_of_target_file:subjName在倒数第几个block内(第一个计数为1)
        # 如'D:\myCodes\workstation_20180829_dynamicFC\FunImgARW\1-500\00002_resting\dti\dic.txt'
        # 的subjName在倒数第3个中
        '''
        self.allSubjName = self.allFilePath
        for i in range(self.unique_id_level_of_target_file - 1):
            self.allSubjName = [os.path.dirname(
                allFilePath_) for allFilePath_ in self.allSubjName]
        self.allSubjName = [os.path.basename(
            allFilePath_) for allFilePath_ in self.allSubjName]
        self.allSubjName = pd.DataFrame(self.allSubjName)
        self.allSubjName_raw = self.allSubjName
        return self

    def fetch_folerNameContainingFile(self):
        '''
        如果file上一级uid不是subject name，那么就涉及到选择那个文件夹下的file
        此时先确定每一个file上面的uid name(可能是模态名)，然后根据你的关键词来筛选
        '''
        self.folerNameContainingFile = [os.path.dirname(
            allFilePath_) for allFilePath_ in self.allFilePath]
        self.folerNameContainingFile = [os.path.basename(
            folderName) for folderName in self.folerNameContainingFile]
        return self

    def fetch_allFileName(self):
        '''
        获取所有file name，用于后续的筛选。
        适用场景：假如跟file一起的有我们不需要的file，
        比如混杂在dicom file中的有text文件，而这些text是我们不想要的。
        '''
        self.allFileName = [os.path.basename(
            allFilePath_) for allFilePath_ in self.allFilePath]
        return self

    # %%  screen according several rules
    def screen_pathLogicalLocation_accordingTo_yourSubjName(self):
        """ 匹配subject name：注意此处用精确匹配，只有完成匹配时，才匹配成功"""
        """maker sure subjName_forSelect is pd.Series and its content is string"""

        if isinstance(self.subjName_forSelect, type(pd.DataFrame([1]))):
            self.subjName_forSelect = self.subjName_forSelect.iloc[:, 0]
        if not isinstance(self.subjName_forSelect[0], str):
            self.subjName_forSelect = pd.Series(
                self.subjName_forSelect, dtype='str')

        # 一定要注意匹配对之间的数据类型要一致！！！
        try:
            # 提取所有被试的uid
            #        self.logic_index_subjname=\
            #                    np.sum(
            #                            pd.DataFrame(
            #                                    [self.allSubjName.iloc[:,0].str.contains\
            #                                    (name_for_self) for name_for_self in self.subjName_forSelect]
            #                                        ).T,
            #                            axis=1)
            #
            #        self.logic_index_subjname=self.logic_index_subjname>=1

            self.allSubjName = self.allSubjName.iloc[:, 0].str.findall(
                self.keyword_of_target_file_uid)

            # 正则表达提取后，可能有的不匹配而为空list,此时应该把空list当作不匹配而去除
            allSubjName_temp = []
            for name in self.allSubjName.values:
                if name:
                    allSubjName_temp.append(name[self.ith_number_of_targetfile_uid])
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
        """ 匹配folerNameContainingFile：注意此处用的连续模糊匹配，只要含有这个关键词，则匹配
        """
        if self.keyword_of_parent_folder_containing_target_file:
            self.logic_index_foler_name_containing_file = [
                self.keyword_of_parent_folder_containing_target_file in oneName_ for oneName_ in self.folerNameContainingFile]
            self.logic_index_foler_name_containing_file = pd.DataFrame(
                self.logic_index_foler_name_containing_file)
        else:
            self.logic_index_foler_name_containing_file = np.ones(
                [len(self.folerNameContainingFile), 1]) == 1
            self.logic_index_foler_name_containing_file = pd.DataFrame(
                self.logic_index_foler_name_containing_file)
        return self

    def screen_pathLogicalLocation_accordingTo_fileName(self):
        """ 匹配file name (不是用于提取uid):正则表达式匹配
        """
        if self.keywork_of_target_file:
            self.allFileName = pd.Series(self.allFileName)
            self.logic_index_file_name = self.allFileName.str.contains(
                self.keywork_of_target_file)
        else:
            self.logic_index_file_name = np.ones([len(self.allFileName), 1]) == 1
            self.logic_index_file_name = pd.DataFrame(self.logic_index_file_name)

        return self

    # %% final logical location of selfected file path
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

    def fetch_selfectedFilePath_accordingPathLogicalLocation(self):
        # source_folder
        self.allFilePath = pd.DataFrame(self.allFilePath)
        self.allSelectedFilePath = self.allFilePath[self.logic_index_all]
        self.allSelectedFilePath = self.allSelectedFilePath.dropna()
        # uid name
        self.allSubjName = pd.DataFrame(self.allSubjName)
        self.allSelectedSubjName = self.allSubjName[self.logic_index_all]
        self.allSelectedSubjName = self.allSelectedSubjName.dropna()
        # raw name
        self.allSubjName_raw = pd.DataFrame(self.allSubjName_raw)
        self.allSelectedSubjName_raw = self.allSubjName_raw[self.logic_index_all]
        self.allSelectedSubjName_raw = self.allSelectedSubjName_raw.dropna()
        return self

    def copy_base(self, i, subjName):
        n_allSelectedSubj = len(np.unique(self.allSelectedSubjName_raw))
        # 每个file保存到每个subjxxx文件夹下面
        if self.save_into_one_or_more_folder == 'one_file_one_folder':
            folder_name = subjName.split('.')[0]
            output_folder = os.path.join(self.out_path, folder_name)
            # 新建subjxxx文件夹
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        # 所有file保存到一个uid下面（file的名字以subjxxx命名）
        elif self.save_into_one_or_more_folder == 'all_files_in_one_folder':
            output_folder = os.path.join(
                self.out_path, subjName + self.save_suffix)
        else:
            print("Please specify how to save: one_file_one_folder OR all_files_in_one_folder")

        # copying OR moving OR do nothing
        fileIndex = self.allSelectedSubjName_raw[(
            self.allSelectedSubjName_raw.values == subjName)].index.tolist()
        if (self.is_copy) and (not self.is_move):
            [shutil.copy(self.allSelectedFilePath.loc[fileIndex_, :][0],
                         output_folder) for fileIndex_ in fileIndex]
        elif (not self.is_copy) and (self.is_move):
            [shutil.move(self.allSelectedFilePath.loc[fileIndex_, :][0],
                         output_folder) for fileIndex_ in fileIndex]
        elif (not self.is_copy) and (not self.is_move):
            print('### No copy and No move ###\n')

        print('Copy the {}/{}th subject: {} OK!\n'.format(i + 1, n_allSelectedSubj, subjName))

    def copy_multiprocess(self):
        s = time.time()
        # 每个file保存到每个subjxxx文件夹下面
        if self.save_into_one_or_more_folder == 'one_file_one_folder':
            pass
        elif self.save_into_one_or_more_folder == 'all_files_in_one_folder':
            pass
        else:
            print(
                "###没有指定复制到一个文件夹还是每个被试文件夹###\n###{}跟'all_files_in_one_folder' OR 'one_file_one_folder'都不符合###".format(
                    self.save_into_one_or_more_folder))

        # 多线程
        # unique的name
        uniSubjName = self.allSelectedSubjName_raw.iloc[:, 0].unique()
        print('Copying...\n')
        cores = multiprocessing.cpu_count()
        if self.n_processess > cores:
            self.n_processess = cores - 1

        with ThreadPoolExecutor(self.n_processess) as executor:
            for i, subjName in enumerate(uniSubjName):
                executor.submit(self.copy_base, i, subjName)

        print('=' * 30)
        #
        e = time.time()
        print('Done!\nRunning time is {:.1f} second'.format(e - s))

    # %%
    def main_run(self):
        # all source_folder and name
        self._after_init()
        self = self.walkAllPath()
        self = self.fetch_allFilePath()
        self = self.fetch_allSubjName()
        self = self.fetch_allFileName()
        # selfect
        self = self.fetch_folerNameContainingFile()
        # logicLoc_subjName：根据被试名字匹配所得到的logicLoc。以此类推。
        # fileName≠subjName,比如fileName可以是xxx.nii,但是subjName可能是subjxxx
        self = self.screen_pathLogicalLocation_accordingTo_yourSubjName()
        self = self.screen_pathLogicalLocation_accordingTo_folerNameContainingFile()
        self = self.screen_pathLogicalLocation_accordingTo_fileName()
        self = self.fetch_totalLogicalLocation()
        self = self.fetch_selfectedFilePath_accordingPathLogicalLocation()

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
                self.out_path))
        print('=' * 50 + '\n')

        # save for checking
        if self.is_save_log:

            # time information
            now = time.localtime()
            now = time.strftime("%Y-%m-%d %H:%M:%S", now)

            # all matched name
            uniSubjName = self.allSelectedSubjName_raw.iloc[:, 0].unique()
            uniSubjName = [uniSubjName_ for uniSubjName_ in uniSubjName]
            uniSubjName = pd.DataFrame(uniSubjName)
            uniSubjName.to_csv(
                os.path.join(
                    self.out_path,
                    'log_allSelectedSubjName.txt'),
                index=False,
                header=False)

            # 所有不匹配的被试名称
            self.unmatched_ref.to_csv(
                os.path.join(
                    self.out_path,
                    'log_unmatched_reference.txt'),
                index=False,
                header=False)

            # 被选路径下所有的文件夹名称
            pd.DataFrame(pd.unique(self.allSubjName.iloc[:, 0])).dropna().to_csv(
                os.path.join(self.out_path, 'log_alltargetfilename.txt'), index=False, header=False)

            # 所有匹配的文件路径
            self.allSelectedFilePath.to_csv(
                os.path.join(
                    self.out_path,
                    'log_allSelectedFilePath.txt'),
                index=False,
                header=False)

            # 保存log
            f = open(
                os.path.join(
                    self.out_path,
                    "log_copy_inputs.txt"),
                'a')
            f.write("\n\n")
            f.write('====================' + now + '====================')
            f.write("\n\n")
            f.write("reference_file is: " + self.reference_file)
            f.write("\n\n")
            f.write(
                "keyword_of_parent_folder_containing_target_file are: " +
                self.keyword_of_parent_folder_containing_target_file)
            f.write("\n\n")
            f.write("unique_id_level_of_target_file is: " +
                    str(self.unique_id_level_of_target_file))
            f.write("\n\n")
            f.write("keyword_of_target_file_uid is: " +
                    str(self.keyword_of_target_file_uid))
            f.write("\n\n")
            f.write("keyword_of_target_file_uid is: " +
                    str(self.keyword_of_target_file_uid))
            f.write("\n\n")
            f.write("targe_file_folder is: " + self.targe_file_folder)
            f.write("\n\n")
            f.write("out_path is: " + self.out_path)
            f.write("\n\n")
            f.write("n_processess is: " + str(self.n_processess))
            f.write("\n\n")
            f.close()

        # copy
        if self.is_run:
            self.copy_multiprocess()
        return self


# %%
if __name__ == '__main__':
    uid = r'D:\WorkStation_2018\WorkStation_dynamicFC_V3\Data\ID_Scale_Headmotion\id.xlsx'
    source_folder = r'D:\WorkStation_2018\WorkStation_dynamicFC_V3\Data\results\windowlength17__silhoutte_and_davies-bouldin\daviesbouldin\metrics'
    out_path = r'D:\WorkStation_2018\WorkStation_dynamicFC_V3\Data\results\windowlength17__silhoutte_and_davies-bouldin\daviesbouldin\610\metrics'
    
    unique_id_level_of_target_file = 1
    keywork_of_target_file = ''
    save_suffix= ''
    
    copy = CopyFmri(
            reference_file=uid,
            targe_file_folder=source_folder,
            keywork_of_reference_uid='([1-9]\d*)',
            ith_number_of_reference_uid=0,
            keyword_of_target_file_uid='([1-9]\d*)',
            ith_number_of_targetfile_uid=0,
            unique_id_level_of_target_file=unique_id_level_of_target_file,
            keywork_of_target_file=keywork_of_target_file,
            keyword_of_parent_folder_containing_target_file='',
            out_path=out_path,
            n_processess=8,
            is_save_log=0,
            is_copy=1,
            is_move=0,
            save_into_one_or_more_folder='all_files_in_one_folder',
            save_suffix=save_suffix,
            is_run=1)
    
    results = copy.main_run()
    # --------------------------------
    results=results.__dict__
    print(results.keys())
    print('Done!')