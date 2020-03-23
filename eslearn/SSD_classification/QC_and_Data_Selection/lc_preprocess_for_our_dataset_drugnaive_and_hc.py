"""
This script is used to pre-process the dataeset (drug naive and first episode and hc) in our center.
1.Transform the .mat files to one .npy file
2. Give labels to each subject, concatenate at the first column
3. Randomly splitting the whole data into training and validation
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import numpy as np 
import pandas as pd
import os
from Utils.lc_read_write_mat import read_mat

# Inputs
matroot = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\SelectedFC_drugnaive_and_hc'  # all mat files directory
scale = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\10-24大表.xlsx'  # whole scale path
n_node = 246  #  number of nodes in the mat network

# Transform the .mat files to one .npy file
allmatpath = os.listdir(matroot)
allmatpath = [os.path.join(matroot, matpath) for matpath in allmatpath]
mask = np.triu(np.ones(n_node),1)==1
allmat = [read_mat(matpath)[mask].T for matpath in allmatpath]
allmat = pd.DataFrame(np.float32(allmat))

# Give labels to each subject, concatenate at the first column
uid = [os.path.basename(matpath) for matpath in allmatpath]
uid = pd.Series(uid)
uid = uid.str.findall('([1-9]\d*)')
uid = pd.DataFrame([np.int(id[0]) for id in uid])
scale = pd.read_excel(scale)
selected_diagnosis = pd.merge(uid, scale, left_on=0, right_on='folder', how='inner')[['folder','诊断']]
selected_diagnosis[selected_diagnosis==1] = 0
selected_diagnosis[selected_diagnosis==3] = 1
allmat_plus_label = pd.concat([selected_diagnosis, allmat],axis=1)
np.random.permutation(255)[:63]
mat_hc = allmat_plus_label.loc[allmat_plus_label[allmat_plus_label['诊断']==0].index[np.random.permutation(255)[:63]]]
allmat_plus_label = pd.concat([allmat_plus_label[allmat_plus_label['诊断'] == 1], mat_hc],axis=0)

# np.save(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_drugnaive_and_hc_from550.npy',allmat_plus_label)
