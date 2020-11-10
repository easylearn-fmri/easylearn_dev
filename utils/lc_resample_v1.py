# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:45:52 2019

@author: Li Chao
"""
import SimpleITK as sitk
import numpy as np
import os

from lc_resample_base import ResampleImg


class ResampleImgV1(ResampleImg):
    """
    Resample a 3D old_image to given new spacing
    The new voxel spacing will determine the new old_image dimensions.
    If is orginal data, use sitk.sitkLinear. 
    If is binary mask, usse sitk.sitkNearestNeighbor
    """
    def __init__(sel,
                 root_roi_path,
                 outpath,
                 datatype,
                 is_overwrite=True):

        super().__init__()
        sel.root_roi_path = root_roi_path
        sel.outpath = outpath
        sel.datatype = datatype
        sel.is_overwrite = is_overwrite
        sel._new_spacing = np.array([0.684, 0.684, 0.684])

    def read_roi_path(sel):
        return [os.path.join(sel.root_roi_path, imgname) for imgname in os.listdir(sel.root_roi_path)]

    def resample_for_allroi(sel):
        allroi = sel.read_roi_path()
        for i, roipath in enumerate(allroi):
            sel.resample_for_oneroi(roipath, is_overwrite=sel.is_overwrite)
        else:
            print('All Done!\n')

    def resample_for_oneroi(sel, roipath, is_overwrite=False):
        #        allroi = sel.read_roi_path()
        #        roipath = allroi[0]
        roiname = os.path.basename(roipath)
        allsubjfile_path = [os.path.join(roipath, roi)
                            for roi in os.listdir(roipath)]
        n_subj = len(allsubjfile_path)
        for i, file in enumerate(allsubjfile_path):
            print(f'{roiname}:{i+1}/{n_subj}...')
            # make save folder to save img file
            if sel.datatype == 'series':
                savefilename = os.path.basename(file) + '.nii'
            else:
                savefilename = os.path.basename(file)

            saveroiname = os.path.basename(os.path.dirname(file))
            savefolder = os.path.join(sel.outpath, saveroiname)
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)

            savename = os.path.join(sel.outpath, saveroiname, savefilename)
            # If img file exists, then overwirte or pass
            status = f'{roiname} failed!'
            if os.path.exists(savename):
                if is_overwrite:
                    print(f'\t{savename} exist Resampling and Overwriting...\n')
                    newimage = sel.resample(
                        file, datatype=sel.datatype)  # resample!
                    sitk.WriteImage(newimage, savename)
                    status = f'{roiname} successfully!'
                else:
                    print(f'\t{savename} exist Pass!\n')
                    continue
            else:
                print('\tResampling and Writting...')
                newimage = sel.resample(
                    file, datatype=sel.datatype)  # resample!
                sitk.WriteImage(newimage, savename)
                status = f'{roiname} successfully!\n'
            print(status)


if __name__ == '__main__':
    sel = ResampleImgV1(root_roi_path=r'I:\Project_Lyph\DICOM\venous_splited\DICOM',
                        outpath=r'I:\Project_Lyph\DICOM\venous_splited\DICOM_resampled_v1',
                        datatype='series',
                        is_overwrite=False)
    sel.resample_for_allroi()
