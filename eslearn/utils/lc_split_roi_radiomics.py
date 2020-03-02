# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:10:18 2019

@author: lenovo
"""
from lc_read_nii import save_nii
from lc_read_nii import read_sigleNii_LC
import nibabel as nib
import numpy as np
import os
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')
from concurrent.futures import ThreadPoolExecutor


class SplitROI():
    """把融合的ROI分成多个单独的ROI，并且保存到不同的文件夹
        Attr:
            all_roi_path:所有融合ROI文件存放的文件夹路径
            out_path:分成单个ROI文件后，保存到哪个路径（代码自动生成亚文件夹，来保存单ROI文件）
        Return:
            将单个ROI文件保存到相应的文件夹下
    """

    def __init__(self, all_roi_path='', out_path=''):
        """所有输入"""
        self.all_roi_path = all_roi_path
        self.out_path = out_path
        self.overcopy = True  # if over copy

        if self.all_roi_path == '':
        	print(f'input path not given')
        	self.my_exit()
        if self.out_path == '':
        	print(f'output path not given')
        	self.my_exit()


    def my_exit(self):
    	sys.exit(1)


    def _read_roi_path(self):
        file_name = os.listdir(self.all_roi_path)
        all_file_path = [
            os.path.join(
                self.all_roi_path,
                filename) for filename in file_name]
        return all_file_path

    def split_roi_for_all_subj(self, all_file_path):
        n_subj = len(all_file_path)
#        with ThreadPoolExecutor(1) as executor:
#            print("Multiprocessing begin...\n")
        for i, file_path in enumerate(all_file_path):
             print("spliting {}/{} subject\n".format(i + 1, n_subj))
#            self.split_roi_for_one_subj(i, n_subj, file_path)
             self.split_roi_for_one_subj(file_path)

    def read(self, nii_name):
        nii_data, nii_object = read_sigleNii_LC(nii_name)
        yield nii_data

    def split_roi_for_one_subj(self, file_path):
        """load nii--split roi--save to """
        # load nii
        nii_name = file_path
        nii_data, nii_object = read_sigleNii_LC(nii_name)
        header = nii_object.header
        affine = nii_object.affine
        
        # split roi
        uni_label = np.unique(nii_data)
        uni_label = list(set(uni_label) - set([0]))  # 去掉0背景
        subjname = os.path.basename(file_path).split('.')[0]
        # split and save to nii
        for label in uni_label:
            # creat folder
            save_folder_name = os.path.join(self.out_path, subjname, 'ROI_' + str(label))
            if not os.path.exists(save_folder_name):
                os.makedirs(save_folder_name)
            save_file_name = os.path.join(
                save_folder_name, os.path.basename(nii_name))
            
            if os.path.exists(save_file_name):
                print('{} exist!\n'.format(save_file_name))
                if self.overcopy:
                    print(f'overwrite!')
                else:
                    continue

            # split
            roi_logic = np.array(nii_data == label, dtype=float)
            roi = nii_data*roi_logic
            # ndarry to nifti object
            roi = nib.Nifti1Image(roi, affine=affine, header=header)

            # save
            save_nii(roi, save_file_name)

    def run(self):
        all_file_path = self._read_roi_path()
        self.split_roi_for_all_subj(all_file_path)


if __name__ == "__main__":
    splitroi = SplitROI(all_roi_path='', out_path=r'I:\Project_Lyph\Raw\Grouped_ROI_Nocontrast_v1')
    splitroi.run()