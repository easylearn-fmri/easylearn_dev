# utf-8
"""
Move or copy roi folder that saving in each subject's folder to the same root folder named ROI$i
"""


import os
import numpy as np
import shutil


def move_roi_to_root_allsubj(root_subjfolder, outpath):
    """
    Move roi to root folder for one subject
    """
    all_subjpath = os.listdir(root_subjfolder)
    all_subjpath = [os.path.join(root_subjfolder, allsub) for allsub in all_subjpath]
    nsubj = len(all_subjpath)
    for i, asp in enumerate(all_subjpath):
        print(f'{i+1}/{nsubj}\n')
        move_roi_to_root_onesubj(asp, outpath)


def move_roi_to_root_onesubj(subjpath, outpath):
    """
    Move roi to root folder for one subject
    """
    # read roi folder path
    roiname = os.listdir(subjpath)
    roipath = [os.path.join(subjpath, rn) for rn in roiname]
    # which folder for saving roi
    uni_roi = np.unique(roiname)
    outsubpath = [os.path.join(outpath, ur) for ur in uni_roi]
    # move subject's roi to root ROI folder
    for rp, osp in zip(roipath, outsubpath):
        filename = os.listdir(rp)
        if len(filename)==0:
            print(f'{rp} containing nothing!')
            continue
        elif len(filename) > 1:
            print(f'{rp} containing multiple files!')
            continue
        else:
            filename = filename[0]   	
            
        # exe  
        if not os.path.exists(osp):
            os.makedirs(osp)
            
        inname = os.path.join(rp, filename)
        outname = os.path.join(osp, filename)
        
        # pass exist file
        if os.path.exists(outname):
            print(f'{outname} exist!')
        else:
            shutil.copy(inname,outname)


if __name__ == '__main__':
	root_subjfolder = r'I:\Project_Lyph\Raw\Grouped_ROI_Nocontrast_v1'
	outpath = r'I:\Project_Lyph\Raw\ROI_Nocontrast_splited_v1'
	move_roi_to_root_allsubj(root_subjfolder, outpath)