# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:26:21 2019
https://www.cnblogs.com/XDU-Lakers/p/9863114.html
@author: lenovo
"""
from SimpleITK import ReadImage, ImageSeriesReader, GetArrayFromImage


def readdcmfile(filename):
    """
    This function used to read information from dicom file.
    Input:
        filename: file path to dicom file.
    Returns:
        spacing, machine, image_array, shape

    Note.: SimpleITK read image in the order of z-y-x, namely the number of slice-width-height;
    However,SimpleITK read origin and spacing in the order of x-y-z.
    """
    image = ReadImage(filename)
    machine = image.GetMetaData('0008|0070')
    manufacturer_model_name = image.GetMetaData('0008|1090') 
    image_array = GetArrayFromImage(image)  # in the order of z, y, x
    shape = image.GetSize()
#    origin = image.GetOrigin()  # in the order of x, y, z
    spacing = image.GetSpacing()  # in the order of x, y, z

    return spacing, machine, manufacturer_model_name, image_array, shape


def readdcmseries(folderpath, get_seriesinfo=True):
    """
    This function used to read information from dicom series folder.
    Input:
        folderpath: path to dicom series folder.
    Returns:
        dicom_names, spacing, machine, image_array, shape, errorseries.
    Note.: SimpleITK read image in the order of z-y-x, namely the number of slice-width-height;
    However,SimpleITK read origin and spacing in the order of x-y-z
    """
    reader = ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folderpath)
    # series info
    if get_seriesinfo:
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        series_ids = reader.GetGDCMSeriesIDs(folderpath)  # get all series id
        series_file_names = reader.GetGDCMSeriesFileNames(folderpath, series_ids[0])  # get the first series
        reader.SetFileNames(series_file_names)
        # extract info
        slice_number = 0  # select the first slice
        try:
            image = reader.Execute()  # type: sitk.Image
            image_array = GetArrayFromImage(image)  # z, y, x
            shape = image.GetSize()
            spacing = image.GetSpacing() # x, y, z
            errorseries = ''
        except RuntimeError:
            image_array = 0  # z, y, x
            shape = 0
            spacing = (0, 0, 0)
            errorseries = folderpath
            print(f'{folderpath} dimension error!\n')
        machine = reader.GetMetaData(slice_number, '0008|0070')
        manufacturer_model_name = reader.GetMetaData(slice_number, '0008|1090') 
#        keys = reader.GetMetaDataKeys(slice_number)  # get all tags key, then we also can get all information
#        info = []
#        for k in keys:
#            print(f'{k} {reader.GetMetaData(0, k)}')  
    else:
        # one dcm info
        spacing, machine, image_array, shape = readdcmfile(dicom_names[0])

    return dicom_names, spacing, machine, manufacturer_model_name, image_array, shape, errorseries


if __name__ == '__main__':
    filename = r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_matlab\Workstation\Decoding the transdiagnostic psychiatric diseases at the dimension level\QC\dcm\liuxiaoyan_Philips.dcm'
    spacing, machine, manufacturer_model_name, image_array, shape = readdcmfile(filename)
    
#    infoname = 'manufacturer_model_name'
#    info = manufacturer_model_name
    
#    import pandas as pd
#    info = pd.DataFrame(info)

