# -*- coding: utf-8 -*-
"""Concatenate multiple csv files into a single excel file
"""

from os import listdir, getcwd
from os.path import join
import pandas as pd
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
dataframe_csv = pd.DataFrame([])
for file in files_csv:
    dataframe_csv = pd.concat([dataframe_csv, pd.read_csv(join(args.dir_csv,file), header=None)], axis=1)
dataframe_csv.to_excel(join(args.dir_out, "combined_csv.xlsx"), header=False, index=False)

input("")