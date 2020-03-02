# utf-8
"""
Author: Li Chao
Email: lichao19870617@gmail.com

For: Radiomics
Useage: 将原始series(underlay, dicomfile)与ROI(mask/overlay)对应起来

INPUT: root_roi and root_origin
    ROI:
        /root_roi
            /06278717_WANGTIEHAN_R03855066_S1_Merge
                /ROI_1
                    /06278717_WANGTIEHAN_R03855066_S1_Merge.nii  # only one file
                /ROI_2
                    /06278717_WANGTIEHAN_R03855066_S1_Merge.nii
                /ROI_n
    
            /07022112_LIZHONGLIANG_R04164650_S1_Merge
                /ROI_1
                /ROI_3
                /ROI_n
    
    Origin:
        /root_origin
            /06278717_WANGTIEHAN_R03855066
                /06278717_WANGTIEHAN_R03855066_S1
                /06278717_WANGTIEHAN_R03855066_S2
                /06278717_WANGTIEHAN_R03855066_Sn
                
            /07022112_LIZHONGLIANG_R04164650
 
----------------------------------------------------------------------------------          
OUTPUT: sorted results according to ROI order
    /savepath
        /ROI
            /ROI_1  # containing all ROI_1 file of all subjects
                /06278717_WANGTIEHAN_R03855066_S1_Merge.nii 
                /06278717_WANGTIEHAN_R03855066_S1_Merge.nii 
                /06278717_WANGTIEHAN_R03855066_S1_Merge.nii 
                /ID_of_subject_n.nii
            /ROI_n
            
        /Origin
            /ROI_1  # containing all ROI_1 series of all subjects that match the ROI
                /06278717_WANGTIEHAN_R03855066_S1
                /06278717_WANGTIEHAN_R03855066_S2
                /06278717_WANGTIEHAN_R03855066_Sn
            /ROI_n
----------------------------------------------------------------------------------

"""

import os
import shutil
import numpy as np

# input
root_roi = r'D:\dms-lymph-nodes\mask'
root_origin = r'D:\dms-lymph-nodes\1_finish'


def get_maskfiles(root_roi=r'D:\dms-lymph-nodes\mask'):
    subjname = os.listdir(root_roi)
    subjpath = [os.path.join(root_roi, name) for name in subjname]
    roiname = [os.listdir(path) for path in subjpath]
    roipath = []
    for path, name in zip(subjpath, roiname):
        roipath.append([os.path.join(path, n) for n in name])
    # flatten
    roipath = flatten_list(roipath)
    # extract uid
    uid_roipath = [mystr.split('\\')[-2] for mystr in roipath]
    uid_roipath = [mystr.split('Merge')[0][0:-1] for mystr in uid_roipath]
    flatten_roiname = np.array([mystr.split('\\')[-1] for mystr in roipath])
    uni_roi = np.unique(flatten_roiname)
    roiname_location = [flatten_roiname == ur for ur in uni_roi]
    return roipath, uni_roi, uid_roipath, roiname_location


def get_originalfiles(root_origin=r'D:\dms-lymph-nodes\1_finish'):
    subjname = os.listdir(root_origin)
    subjpath = [os.path.join(root_origin, name) for name in subjname]
    seriesname = [os.listdir(path) for path in subjpath]
    seriespath = []
    for path, name in zip(subjpath, seriesname):
        seriespath.append([os.path.join(path, n) for n in name])
    # flatten
    seriespath = flatten_list(seriespath)
    # extract uid
    uid_seriespath = [mystr.split('\\')[-1] for mystr in seriespath]
    return seriespath, uid_seriespath


def flatten_list(mylist):
    return [item for sublist in mylist for item in sublist]


def match(roipath, uid_roipath, seriespath, uid_seriespath, roiname_location):
    uid_roipath_for_eachroi = [
        np.array(uid_roipath)[logic] for logic in roiname_location]
    matched_series_location = []
    for uid_roi in uid_roipath_for_eachroi:
        matched_series_location.append(
            [np.where(np.array(uid_seriespath) == aa, True, False) for aa in uid_roi])
    return matched_series_location


def move(savepath, uni_roi, roipath, seriespath, roiname_location, matched_series_location):
    count = 1
    nsubj = len(uni_roi)
    for rn, rl in zip(uni_roi, roiname_location):
        #%% ROI
        # create folder to save results
        newfolder = os.path.join(savepath, 'ROI', rn)
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)

        # move
        oldfolder = np.array(roipath)[rl]
        # because one ROI folder only one file, so [0]
        oldfile = [os.listdir(folder)[0] for folder in oldfolder]
        oldfilepath = [os.path.join(folder, file)
                       for folder, file in zip(oldfolder, oldfile)]
        newfilepath = [os.path.join(newfolder, file) for file in oldfile]

        count_inner = 1
        nfile = len(newfilepath)
        for old, new in zip(oldfilepath, newfilepath):
            print(
                f'Running for ROI {count}/{nsubj} [subprocessing {count_inner}/{nfile}] ...\n')
            if os.path.exists(new):
                print(f'{new} exists!\n')
            else:
                shutil.copy(old, new)
            count_inner += 1
        count += 1
    else:
        print(f'Processing ROI completed!\n')

    #%% origin
    count = 1
    for rn, sl in zip(uni_roi, matched_series_location):
        # create folder to save results
        newfolder = os.path.join(savepath, 'Origin', rn)
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)
        # move
        oldfolder = [np.array(seriespath)[asl][0] for asl in sl]
        newsubfolder = [os.path.join(
            newfolder, os.path.basename(old)) for old in oldfolder]

        count_inner = 1
        nfile = len(newsubfolder)
        for old, new in zip(oldfolder, newsubfolder):
            print(
                f'Running for Origin {count}/{nsubj} [subprocessing {count_inner}/{nfile}] ...\n')
            if os.path.exists(new):
                print(f'{new} exists!\n')
            else:
                shutil.copytree(old, new)
            count_inner += 1

        count += 1
    else:
        print(f'Processing Origin completed!\n')


#%%
if __name__ == '__main__':
	"""
	python lc_match_maskVSorigin.py I:\\Project_Lyph\\ROI_venus D:\\dms-lymph-nodes\\1_finish I:\\Project_Lyph\\Grouped_ROI_venous
	"""
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('root_roi', type=str, help='ROI文件夹')
	parser.add_argument('root_origin', type=str, help='原始文件夹')
	parser.add_argument('savepath', type=str, help='保存结果的路径')
	args = parser.parse_args()

	#%%
	roipath, uni_roi, uid_roipath, roiname_location = get_maskfiles(root_roi=args.root_roi)

	seriespath, uid_seriespath = get_originalfiles( root_origin=args.root_origin)

	matched_series_location = match(roipath, uid_roipath, seriespath, uid_seriespath, roiname_location)

	move(args.savepath, uni_roi, roipath, seriespath, roiname_location, matched_series_location)
