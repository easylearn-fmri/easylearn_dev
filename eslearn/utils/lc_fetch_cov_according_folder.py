# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:04:33 2019
当给定了影像数据和量表时，如果量表数据包括而且大于影像数据时，我们需要从中提取与影像数据匹配的部分
@author: lenovo
"""
import sys
import os
cpwd = __file__
root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)
print(f'##{root}')
import pandas as pd
import Utils.lc_copy_selected_file_V6 as copy


class screening_covariance_to_match_neuroimage():

    def __init__(sel):
        sel.folder = r'D:\WorkStation_2018\WorkStation_dynamicFC_V1\Data\zDynamic\state\covariances\folder_MDD.xlsx'
        sel.path_neuroimage = r'D:\WorkStation_2018\WorkStation_dynamicFC_V1\Data\zDynamic\state\allState17_5\state5_all\state5\state5_MDD'
        sel.cov_path = r'D:\WorkStation_2018\WorkStation_dynamicFC_V1\Data\zDynamic\state\covariances\ageANDsex_MDD.xlsx'
        sel.save_path = r'D:\WorkStation_2018\WorkStation_dynamicFC_V1\Data\zDynamic\state\allState17_5\state5_all\state5\cov'
        sel.save_name = 'state5_cov_MDD.xlsx'

    def fetch_folder(sel):
        """ fetch sel.folder"""

        sel_folder = copy.CopyFmri(
                reference_file=sel.folder,
                targe_file_folder=sel.path_neuroimage,
        
                keywork_reference_for_uid='([1-9]\d*)',
                ith_reference_for_uid=0,
                keyword_targetfile_for_uid='([1-9]\d*)',
                matching_pointnumber_in_backwards=1,
                ith_targetfile_for_uid=0,
        
                keyword_targetfile_not_for_uid='',
                keyword_parentfolder_contain_targetfile='',
        
                savePath=sel.save_path,
                n_processess=2,
                ifSaveLog=0,
                ifCopy=0,
                ifMove=0,
                saveInToOneOrMoreFolder='saveToOneFolder',
                saveNameSuffix='',
                ifRun=0)

        result = sel_folder.main_run()

        uid = result.allSubjName
        values = [int(v) for v in uid.values]
        uid = pd.DataFrame(values)

        return uid

    def fecth_cov_acord_to_folder(sel,uid, left_on=0, right_on='Unnamed: 0'):
        """求folder和cov的交集"""
    
        cov = pd.read_excel(sel.cov_path)
        cov_selected = pd.merge(
            uid,
            cov,
            left_on=left_on,
            right_on=right_on,
            how='inner')
        
        return cov_selected


if __name__ == "__main__":
    sel = screening_covariance_to_match_neuroimage()
    uid = sel.fetch_folder()
    cov_selected = sel.fecth_cov_acord_to_folder(uid)

    # save
    cov_selected[['年龄', '性别']].to_excel(os.path.join(
        sel.save_path, sel.save_name), index=False, header=False)
