# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 09:36:59 2018
read nifti file or files
@author: LiChao
"""
# import
import nibabel as nib
import os
import pandas as pd

# input

class NiiProcessor():
    """
    This class is used to read/save nifti file or files
    """
    def __init__(self):
        pass
    
    def read_sigle_nii(self, img_path):
        img_path = os.path.normpath(img_path)
        # read
        img_object = nib.load(img_path)
        img_data = img_object.get_fdata()
        return img_data, img_object
    
    
    def read_multi_nii(self, img_folder, suffix='.nii'):
        """load multiple image

        When files are img/hdr format, function will delete duplicates.
        """

        img_name = os.listdir(img_folder)
        img_name_uni = [os.path.splitext(name)[0] for name in img_name]
        img_name_uni = pd.DataFrame(img_name_uni).drop_duplicates()
        img_path = [os.path.join(img_folder, ''.join([imgname, suffix]))
                    for imgname in img_name_uni[0]]
    
        img_data = list()
        for path in img_path:
            data, _ = self.read_sigle_nii(path)
            img_data.append(data)
        return img_data, img_name
    
    
    def save_nii(self, img, save_filename):
        """save data to nii"""

        img.to_filename(save_filename)
    #    print (img)
    #    print (img.header['db_name']) 
    
    
    def main(self, img_folder, suffix='.nii'):
        img_data = self.read_multi_nii(img_folder, suffix)
        return img_data


if __name__ == '__main__':
#    img_folder = (r'I:\181127_ForWangFei\ALFF')
#    img_folder1=(r'D:\WorkStation_2018\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\C_Weighted_selected')
#    img_name=('D:\myMatlabCode\Python\zDegreeCentrality_PositiveWeightedSumBrainMap_sub001.nii')
#    img_data, _ = main(img_folder, '.img')
    img_data, img_object = NiiProcessor.read_sigle_nii(r'D:\workstation_b\mReHoMap_sub33.nii')
