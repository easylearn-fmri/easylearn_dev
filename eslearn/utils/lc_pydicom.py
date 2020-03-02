# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:02:42 2019

@author: lenovo
"""

import pydicom
import nibabel
import SimpleITK as sitk


def read_nii(filename):
    nii = nibabel.nifti1.load(filename)
    dataset = nii.get_data()
    return dataset


def nii2dcm(dataset, filename):
    """
    write dicom filename
    filename = 'test.dcm'
    """
    # TODO transfer array to dicom directly
    pydicom.write_file(filename, dataset)

if __name__ == '__main__':
    filename = r"D:\dms\13567701_CHENSHUANG_R03509555\13567701_CHENSHUANG_R03509555_S1\1_2_392_200036_9116_2_5_1_37_2420762357_1456450872_373480_S1_I1.dcm"
    dataset = pydicom.dcmread(filename, force=True) 
    val = dataset.data_element('Columns').value 
    
    image = sitk.ReadImage(filename) 
    image_array = sitk.GetArrayFromImage(image)
    
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()  
    reader.GetMetaData('0008|0008')  
    reader.GetMetaDataKeys()
    nii  = read_nii(filename)

    