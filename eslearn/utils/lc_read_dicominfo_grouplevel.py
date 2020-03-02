# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:30:50 2019

@author: lenovo
"""
import sys
import os
homedir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(homedir)
from os import listdir
from os.path import join
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from numpy import column_stack, row_stack, savetxt

from Utils.lc_read_dicominfo_base import readdcmseries


def run(subjpath, get_seriesinfo):
    """
    For several series folder of one subject
    Input:
        subjpath: one subject's folder path (containing several seriers subfolders)
    """
    print(f'running {subjpath}...')
    allsubj = listdir(subjpath)
    allseriespath = [join(subjpath, subj) for subj in allsubj]
    spacing, machine, seriesname, shape, errorseries = [], [], [], [], []
    for seriespath in allseriespath:
        _, s, m, _,  shp, es = readdcmseries(seriespath, get_seriesinfo)
        spacing.append(s)
        machine.append(m)
        shape.append(shp)
        errorseries.append(es)
        seriesname.append(seriespath)

    return spacing, machine, seriesname, shape, errorseries


def run_all(rootdir, get_seriesinfo, n_process):
    # 多线程
    allsubj = listdir(rootdir)
    allsubjpath = [join(rootdir, subj) for subj in allsubj]

    cores = multiprocessing.cpu_count()
    if n_process > cores:
        n_process = cores - 1

    dcminfo = []
    with ThreadPoolExecutor(n_process) as executor:
        for subjpath in allsubjpath:
            info = executor.submit(run, subjpath, get_seriesinfo)
            dcminfo.append(info.result())
    return dcminfo

def main():
#    python lc_readcminfo_forradiomics.py -rd D:\\dms-lymph-nodes\\test -gs True -np 3 -op D:/dms-lymph-nodes/dcminfo_finish1
	# input
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--rootdir', type=str, help='所有被试根目录')
    parser.add_argument('-gs', '--get_seriesinfo', type=str, help='是否提取dcm series信息')
    parser.add_argument('-np', '--n_process', type=int, help='多核运行数')
    parser.add_argument('-op', '--outpath', type=str, help='保存结果的路径')
    args = parser.parse_args()

    # run
    dcminfo = run_all(rootdir = args.rootdir,  get_seriesinfo=args.get_seriesinfo, n_process=args.n_process)
    
    dcminfo_list = [list(di) for di in dcminfo]
    errorseries = [dl[3] for dl in dcminfo_list]  # dimension error series
    dcminfo_df = [column_stack((dl[2], dl[1], dl[0])) for dl in dcminfo_list]
    dcminfo_alldf = row_stack(dcminfo_df)
    
    # save
    savetxt(join(args.outpath, 'errorseries.txt'), errorseries, fmt='%s', delimiter=' ')
    savetxt(join(args.outpath, 'dcminfo.txt'),
            dcminfo_alldf, fmt='%s', delimiter=',',
            header='series_name, machine, spacing_x, spacing_y, slice_thickness',
            comments='')


if __name__ == '__main__':
#    main()
    
#     for debug
    rootdir = r'D:\dms'
    dcminfo = run_all(rootdir, True, 4)
#    dcminfo = run_all(r'D:\dms-lymph-nodes\test', True, 2)
#    outpath = r'D:\dms-lymph-nodes'
#    
#    dcminfo_list = [list(di) for di in dcminfo]
#    errorseries = [dl[3] for dl in dcminfo_list]  # dimension error series
#    dcminfo_df = [column_stack((dl[2], dl[1], dl[0]))
#                  for dl in dcminfo_list]
#    dcminfo_alldf = row_stack(dcminfo_df)
#    
#    # save
#    savetxt(join(outpath, 'errorseries.txt'), errorseries, fmt='%s', delimiter=' ')
#    savetxt(join(outpath, 'dcminfo.txt'),
#            dcminfo_alldf, fmt='%s', delimiter=',',
#            header='series_name, machine, spacing_x, spacing_y, slice_thickness',
#            comments='')
