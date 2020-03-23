"""
This script is used to transform the 206 dataset into .npy format.
1.Transform the .mat files to one .npy file
2. Give labels to each subject, concatenate at the first column
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import numpy as np 
import pandas as pd
import os
from eslearn.utils.lc_read_write_Mat import read_mat

# Inputs
matroot = r'D:\WorkStation_2018\SZ_classification\Data\SelectedFC_206'  # all mat files directory
scale = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\SZ_NC_108_100.xlsx'  # whole scale path
n_node = 246  #  number of nodes in the mat network

# Transform the .mat files to one .npy file
allmatpath = os.listdir(matroot)
allmatpath = [os.path.join(matroot, matpath) for matpath in allmatpath]
mask = np.triu(np.ones(n_node),1)==1
allmat = [read_mat(matpath)[mask].T for matpath in allmatpath]
allmat = pd.DataFrame(np.float32(allmat))

# Give uid and labels to each subject, concatenate at the first column
uid = [os.path.basename(matpath) for matpath in allmatpath]
uid = pd.Series(uid)
uid = uid.str.findall('(NC.*[0-9]\d*|SZ.*[0-9]\d*)')
uid = [str(id[0]) for id in uid]
uid = pd.DataFrame([''.join(id.split('_')) for id in uid])

scale = pd.read_excel(scale)
selected_diagnosis = pd.merge(uid, scale, left_on=0, right_on='ID', how='inner')[['ID','group']]
selected_diagnosis['group'][selected_diagnosis['group']==2] = 0

allmat_plus_label = pd.concat([selected_diagnosis, allmat],axis=1)

allmat_plus_label['ID'] = allmat_plus_label['ID'].str.replace('NC','10');
allmat_plus_label['ID'] = allmat_plus_label['ID'].str.replace('SZ','20');
allmat_plus_label['ID'] = np.int32(allmat_plus_label['ID'])
np.save(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_206.npy',allmat_plus_label)

#%% Extract covariances: age and sex
cov = pd.merge(uid, scale, left_on=0, right_on='ID', how='inner')[['ID','group', 'age', 'sex']]
cov[['ID', 'group']] = allmat_plus_label[['ID', 'group']]
cov.columns = ['folder', 'diagnosis', 'age', 'sex']
cov.to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\cov_206.txt', index=False)

