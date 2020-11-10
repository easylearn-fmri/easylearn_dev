# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:45:52 2019
@author: lenovo
"""
from SimpleITK import ReadImage, ImageSeriesReader, GetArrayFromImage
import SimpleITK as sitk
import numpy as np


class ResampleImg():
    """
    Resample a 3D old_image to given new spacing
    The new voxel spacing will determine the new old_image dimensions.
    If is orginal data, use sitk.sitkLinear. 
    If is binary mask, usse sitk.sitkNearestNeighbor
	"""
    def __init__(sel):
        sel._new_spacing = np.array([0.684, 0.684, 0.684])

    def resample(sel, old_image_path, datatype='series'):
        """
        Usage: resample(sel, old_image_path)
        Resample a 3D old_image to given new spacing
        The new voxel spacing will determine the new old_image dimensions.
        
        interpolation选项 	所用的插值方法
        INTER_NEAREST 	    最近邻插值
        INTER_LINEAR 	    双线性插值（默认设置）
        INTER_AREA 	        使用像素区域关系进行重采样。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。 但是当图像缩放时，它类似于INTER_NEAREST方法。
        INTER_CUBIC 	    4x4像素邻域的双三次插值
        INTER_LANCZOS4 	    8x8像素邻域的Lanczos插值
        """   
        # read dicom series
        if datatype == 'series':
            reader = ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(old_image_path)
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            series_ids = reader.GetGDCMSeriesIDs(old_image_path)  # get all series id
            series_file_names = reader.GetGDCMSeriesFileNames(old_image_path, series_ids[0])  # get the first series
            reader.SetFileNames(series_file_names)
            old_image = reader.Execute()  # type: sitk.Image
        elif datatype == 'nii':
            # read nifiti file
            old_image = ReadImage(old_image_path)
        else:
            print(f'Datatype {datatype} is wrong!\n')
        
        #  get old information and new information
        old_spacing = old_image.GetSpacing()
        size = old_image.GetSize()
        new_size = (np.round(size*(old_spacing/sel._new_spacing))).astype(int).tolist()
        
        # EXE
        # If is orginal data ('series'), use sitk.sitkLinear. 
        # If is binary mask ('nii'), usse sitk.sitkNearestNeighbor.
        # TODO: other methods;
        # FIXME: Some cases the 'series' may not indicate the orginal data
        # FIXME:Some cases the 'nii' may not indicate the binary mask 
        if datatype == 'series':
            resampled_img = sitk.Resample(old_image, new_size, sitk.Transform(),
            sitk.sitkLinear, old_image.GetOrigin(), sel._new_spacing,
                                  old_image.GetDirection(), 0.0, old_image.GetPixelID())
        elif datatype == 'nii':
            resampled_img = sitk.Resample(old_image, new_size, sitk.Transform(),
            sitk.sitkNearestNeighbor, old_image.GetOrigin(), sel._new_spacing,
                                  old_image.GetDirection(), 0.0, old_image.GetPixelID())
    
    
    #    resampled_img.GetSpacing()
    #    resampled_img.GetSize()  
        return resampled_img


if __name__ == '__main__':
    sel = ResampleImg()
#    resampled_img = sel.resample(old_img_path)

#    resampled_img.GetSpacing()
#    resampled_img.GetSize()
