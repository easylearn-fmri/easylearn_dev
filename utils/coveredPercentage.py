# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:10:02 2018

@author: lenovo
"""
from lc_read_nii import read_sigleNii_LC
import sys
sys.path.append(r'D:\myCodes\LC_MVPA\Python\MVPA_Python\utils')

img_path_3d = r'D:\其他\陈逸凡\aLL_mask3D.nii'
img_path_alff = r'D:\其他\陈逸凡\all_maskALFF.nii'
data_3d = read_sigleNii_LC(img_path_3d)
data_alff = read_sigleNii_LC(img_path_alff)
