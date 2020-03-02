# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:10:18 2019

@author: lenovo
"""
import sys
sys.path.append(r'F:\黎超\dynamicFC\Code\lc_rsfmri_tools_python-master\Utils')
import os
import numpy as np
import nibabel as nib
import shutil

from lc_read_nii import read_sigleNii_LC
from lc_read_nii import save_nii


class GroupSeries():
    """group dcm series to each series folder
        Attr:
            all_subj_path:所有dicom文件存放的文件夹路径
            out_path:group 成各个series后，保存到哪个路径（代码自动生成亚文件夹，来保存单series文件）
        Return:
            NO return, only perform grouped dcm

        Original directory like this: root/subj$i/all_dcm.dcm
        Output diretory like this: root/subj&i/uniqueID_S$i/all_dcm.dcm
    """

    def __init__(sel, all_subj_path=r'D:\dms-lymph-nodes\1_finish',
                 out_path=r'D:\dms-lymph-nodes\1_finish'):
        sel.all_subj_path = all_subj_path
        sel.out_path = out_path

    def read_roi_path(sel):
        sel.subj_name = os.listdir(sel.all_subj_path)
        sel.all_subj_path = [os.path.join(
            sel.all_subj_path, filename) for filename in sel.subj_name]
        return sel

    def group_series_for_all_subj(sel):
        n_subj = len(sel.all_subj_path)
        num_proc = np.arange(0, n_subj, 1)
        for i, file_path, subjname in zip(num_proc, sel.all_subj_path, sel.subj_name):
            print("Grouping {}/{} subject\n".format(i+1, n_subj))
            sel.group_series_for_one_subj(file_path, subjname)

    def group_series_for_one_subj(sel, file_path, subjname):
        """
        load dcm--split dcm according series name--save to 
        """
        # load dcm
        dcm_name = os.listdir(file_path)
        # exclude folder, only include file
        dcm_name = np.array([dn for dn in dcm_name if len(dn.split('.')) > 1])

        dcm_path = np.array([os.path.join(file_path, dn) for dn in dcm_name])
        s_name = [dcm.split('_')[-2] for dcm in dcm_name]
        # grroup dcm
        uni_sname = np.unique(s_name)
        s_logic = [np.where(np.array(s_name) == sn) for sn in uni_sname]

        # split and save
        for loc, sname in zip(s_logic, uni_sname):
            # split
            sdcmpath = dcm_path[loc]
            sdcmname = dcm_name[loc]
            # creat Sn folder
            save_folder_name = os.path.join(
                sel.out_path, subjname, subjname + '_' + sname)
            if not os.path.exists(save_folder_name):
                os.mkdir(save_folder_name)
            # save
            for dp, dn in zip(sdcmpath, sdcmname):
                try:
                    shutil.move(dp, os.path.join(save_folder_name, dn))
                except FileNotFoundError:
                    print('{} may be moved'.format(dp))

    def main(sel):
        sel.read_roi_path()
        sel.group_series_for_all_subj()


if __name__ == "__main__":
    sel = GroupSeries()
    print(sel.all_subj_path)
    # sel.main()
