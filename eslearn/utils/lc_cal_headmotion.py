# -*- coding: utf-8 -*-
"""
extract mean Power mean FD
Combine all subjects' mean FD to one excel file
"""
import numpy as np
import pandas as pd
import os
import re


# class GetPath():
#     """get all files path
#     """
#     def __init__(sel,
#                  root_path=r'I:\Data_Code\Doctor\RealignParameter',
#                  keyword='Van'):

#         sel.root_path = root_path
#         sel.keyword = keyword
#         print("GetPath initiated!")

#     def get_all_subj_path(sel):
#         all_subj = os.listdir(sel.root_path)
#         sel.all_subj_path = [os.path.join(
#             sel.root_path, allsubj) for allsubj in all_subj]

#     def get_all_file_path(sel):
#         sel.all_file_name = [os.listdir(subj_path)
#                              for subj_path in sel.all_subj_path]

#     def screen_file_path(sel):
#         """only select Power PD
#         """
#         file_path = []
#         for i, filename in enumerate(sel.all_file_name):
#             selected_file = [name for name in filename if sel.keyword in name]
#             if selected_file:
#                 selected_file = selected_file[0]
#                 file_path.append(os.path.join(
#                     sel.all_subj_path[i], selected_file))

#         sel.all_file_path = file_path

#     def run_getpath(sel):
#         sel.get_all_subj_path()
#         sel.get_all_file_path()
#         sel.screen_file_path()


# class CalcMeanValue(GetPath):
#     """
#     calculate the mean value (such as mean FD or mean rotation of head motion) for each subject
#     """
#     def __init__(sel):
#         super().__init__(sel)
#         sel.root_path = r'I:\Data_Code\Doctor\RealignParameter'
#         sel.keyword = 'Power'

#         print("CalcMeanValue initiated!")

#     def calc(sel):
#         print("\ncalculating mean value...")
#         sel.MeanValue = [np.loadtxt(file_path).mean(axis=0)  # each column is one parameter
#                          for file_path in sel.all_file_path]
#         print("\ncalculate mean value Done!")

#     def run_calc(sel):
#         sel.calc()


# class SaveMeanFDToEachSubjFolder(CalcMeanValue):
#     def __init__(sel,
#                  savename=r'D:\WorkStation_2018\WorkStation_dynamicFC\Scales\mean6.xlsx'):

#         super().__init__()
#         sel.root_path = r'I:\Data_Code\Doctor\RealignParameter'
#         sel.keyword = 'rp_'
#         sel.savename = savename
#         print("SaveMeanFDToEachSubjFolder initiated!")

#     def _get_subjname(sel, reg='([1-9]\d*)', ith=0):
#         # ith: when has multiple match, select which.
#         path = [os.path.dirname(path) for path in sel.all_file_path]
#         subjname = [os.path.basename(pth) for pth in path]
#         if reg:
#             if ith != None:
#                 sel.subjname = [re.findall(reg, name)[ith]
#                                 for name in subjname]
#             else:
#                 sel.subjname = [re.findall(reg, name) for name in subjname]

#     def combine(sel):
#         """combine mean FD and subjname into DataFrame
#         """
#         mfd = pd.DataFrame(sel.MeanValue)
#         sjnm = pd.DataFrame(sel.subjname)

#         sel.subjname_meanFD = pd.concat([sjnm, mfd], axis=1)
#         sel.subjname_meanFD.columns = ['ID'] + \
#             ['meanvalue' + (str(i + 1)) for i in range(np.shape(mfd)[1])]

#     def save(sel):
#         sel.subjname_meanFD.to_excel(sel.savename, index=False)

#     def run_save(sel):
#         sel._get_subjname()
#         sel.combine()
#         sel.save()


# if __name__ == "__main__":
#     sel = SaveMeanFDToEachSubjFolder()
#     sel.run_getpath()
#     sel.run_calc()
#     sel.run_save()
#     print("\nAll Done!")
