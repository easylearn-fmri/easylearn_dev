# -*- coding: utf-8 -*-
# 
"""Concatenate multiple csv files into a single excel file

Usage: python el_combine_csv2excel.py $directory_of_csv -do $ directory_of_output
"""

from os import listdir, getcwd
from os.path import join
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Inputs')
parser.add_argument('dir_csv',  type=str, help='Please enter a directory containing all csv files')
parser.add_argument('-do', '--dir_out', type=str, default=getcwd(), help='Please enter a output directory')
args = parser.parse_args()

#========All input are here==========
# dir_csv = 'D:/workstation_b/csv'
# dir_out = 'D:/workstation_b'
#====================================

files_csv = listdir(args.dir_csv)
files_csv_df = pd.Series(files_csv)
sub_id = files_csv_df.str.findall('([1-9]\d*)')
sub_id = [np.int16(id[0]) if id != [] else 999999 for id in sub_id]
order_sub_id = np.argsort(sub_id)
sorted_files_csv_df = files_csv_df[order_sub_id]

dataframe_csv = pd.DataFrame([])
for file in sorted_files_csv_df:
    dataframe_csv = pd.concat([dataframe_csv, pd.read_csv(join(args.dir_csv,file), header=None)], axis=1)

dataframe_csv.columns = [cn.split('.')[0] for cn in sorted_files_csv_df]
dataframe_csv.to_excel(join(args.dir_out, "combined_csv.xlsx"), index=False)

input("Done!\n")