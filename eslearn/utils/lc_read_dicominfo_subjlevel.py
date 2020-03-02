# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:30:50 2019
This script is used to read dicom informations from one subject which contains several series folders.
@author: lenovo
"""
import sys
import os
import  numpy
homedir = os.path.dirname(os.getcwd())
sys.path.append(homedir)
from concurrent.futures import ThreadPoolExecutor


from Utils.lc_read_dicominfo_base import readdcmseries


def run(subjpath, get_seriesinfo):
    """
    For several series folder of one subject
    Input:
        subjpath: one subject's folder path (containing several seriers subfolders)
    Returns:
        spacing, machine, seriesname, shape, errorseries
    """
    
    allsubj = os.listdir(subjpath)
    allseriespath = [os.path.join(subjpath, subj) for subj in allsubj]
    spacing, machine, seriesname, shape, errorseries = [], [], [], [], []
    n_subj = len(allseriespath)
    for i, seriespath in enumerate(allseriespath):
        print(f'running {i+1}/{n_subj} \n: {seriespath}...')
        _, s, m, _,  shp, es = readdcmseries(seriespath, get_seriesinfo)
        spacing.append(s)
        machine.append(m)
        shape.append(shp)
        errorseries.append(es)
        seriesname.append(seriespath)

    return spacing, machine, seriesname, shape, errorseries

def main():
#    python lc_readcminfo_forradiomics_onefolder.py -sp I:\\Project_Lyph\\Grouped_ROI_venous\\Origin\\ROI_1 -gs True -op I:\\Project_Lyph\\Grouped_ROI_venous\\Origin\\ROI_1
	# input
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--subjpath', type=str, help='所有被试根目录')
    parser.add_argument('-gs', '--get_seriesinfo', type=str, help='是否提取dcm series信息')
    parser.add_argument('-op', '--outpath', type=str, help='保存结果的路径')
    args = parser.parse_args()

    # run
    spacing, machine, seriesname, shape, errorseries = run(args.subjpath, args.get_seriesinfo)
    
    dcminfo = numpy.column_stack([seriesname, machine, spacing, shape])
    
    # save
    numpy.savetxt(os.path.join(args.outpath, 'errorseries.txt'), errorseries, fmt='%s', delimiter=' ')
    numpy.savetxt(os.path.join(args.outpath, 'dcminfo.txt'),
            dcminfo, fmt='%s', delimiter=',',
            header='series_name, machine, spacing_x, spacing_y, slice_thickness, x, y , z',
            comments='')
    print('Done!')

if __name__ == '__main__':
    subjpath = r'D:\dms\13567701_CHENSHUANG_R03509555'
    spacing, machine, seriesname, shape, errorseries = run(subjpath, True)
    
#     for debug
#    rootdir = r'I:\Project_Lyph\Grouped_ROI_venous\Origin\ROI_1'
#    dcminfo = run_all(r'D:\dms-lymph-nodes\test', True, 2)
#    outpath = r'D:\dms-lymph-nodes'
#    
#    dcminfo_list = [list(di) for di in dcminfo]
#    errorseries = [dl[3] for dl in dcminfo_list]  # dimension error series
#    dcminfo_df = [numpy.column_stack((dl[2], dl[1], dl[0]))
#                  for dl in dcminfo_list]
#    dcminfo_alldf = numpy.row_stack(dcminfo_df)
#    
#    # save
#    numpy.savetxt(os.path.join(outpath, 'errorseries.txt'), errorseries, fmt='%s', delimiter=' ')
#    numpy.savetxt(os.path.join(outpath, 'dcminfo.txt'),
#            dcminfo_alldf, fmt='%s', delimiter=',',
#            header='series_name, machine, spacing_x, spacing_y, slice_thickness',
#            comments='')
