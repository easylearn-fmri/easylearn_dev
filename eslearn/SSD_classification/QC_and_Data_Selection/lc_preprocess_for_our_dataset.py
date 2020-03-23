"""
This script is used to pre-process the dataeset in our center.
1.Transform the .mat files to one .npy file
2. Give labels to each subject, concatenate at the first column
3. Randomly splitting the whole data into training and validation
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import numpy as np 
import pandas as pd
import os
from eslearn.utils.lc_read_write_Mat import read_mat

# Inputs
matroot = r'D:\WorkStation_2018\SZ_classification\Data\SelectedFC550'  # all mat files directory
scale = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'  # whole scale path
uid_unmedicated_and_firstepisode = r'D:\WorkStation_2018\SZ_classification\Scale\uid_unmedicated_and_firstepisode.txt'
uid_sz_chronic_drugtaking_morethan6mon = r'D:\WorkStation_2018\SZ_classification\Scale\精分-非首发用药-病程大于6月.txt'
n_node = 246  #  number of nodes in the mat network

#%% Transform the .mat files to one .npy file
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
age_sex = pd.merge(uid, scale, left_on=0, right_on='folder', how='inner')[['folder', '诊断', '年龄','性别']]

#  Giving large label to SZ
selected_diagnosis[selected_diagnosis==1] = 0
selected_diagnosis[selected_diagnosis==3] = 1
allmat_plus_label = pd.concat([selected_diagnosis, allmat],axis=1)
# print(allmat_plus_label)
#np.save(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_550.npy',allmat_plus_label)

#%% Extract validation dataset that contains first episode unmedicated patients
# unmedicated
uid_unmedicated_and_firstepisode = pd.read_csv(uid_unmedicated_and_firstepisode, header=None)
data_unmedicated_and_firstepisode_550 = allmat_plus_label[allmat_plus_label['folder'].isin(uid_unmedicated_and_firstepisode[0])]
cov_unmedicated_and_firstepisode = age_sex[age_sex['folder'].isin(uid_unmedicated_and_firstepisode[0])] 

# HC: matching hc and sz
from scipy.stats import ttest_ind
from eslearn.statistics.lc_chisqure import lc_chisqure
cov_hc_for_matching_unmedicated_and_firstepisode = age_sex[age_sex['诊断'] == 1] 
np.random.seed(11)
idx_rand = np.random.permutation(len(cov_hc_for_matching_unmedicated_and_firstepisode))
cov_hc = cov_hc_for_matching_unmedicated_and_firstepisode.iloc[idx_rand[:len(cov_unmedicated_and_firstepisode)],:]

# Check if matching
ttest_ind(cov_unmedicated_and_firstepisode['年龄'], cov_hc['年龄'])
lc_chisqure([44, 44], [np.sum(cov_unmedicated_and_firstepisode['性别'] == 1), np.sum(cov_hc['性别'] == 1)])

# Get data and save
data_hc_for_matching_unmedicated_and_firstepisode_550 = allmat_plus_label[allmat_plus_label['folder'].isin(cov_hc['folder'])]
data_all = np.concatenate([data_unmedicated_and_firstepisode_550, data_hc_for_matching_unmedicated_and_firstepisode_550])
# np.save(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_unmedicated_and_firstepisode_550.npy', data_all)

#%% Generate demographic table for Unmedicated and the matching HC
uid_unmedicated_file = r'D:\WorkStation_2018\SZ_classification\Scale\uid_unmedicated_and_firstepisode.txt'
uid_unmedicated = pd.read_csv(uid_unmedicated_file, header=None, dtype=np.int32)
uid_unmedicated_sz_hc = pd.concat([cov_hc['folder'], uid_unmedicated])
scale_unmedicated_hc = pd.merge(scale, uid_unmedicated_sz_hc, left_on='folder', right_on=0, how='inner')[['folder', '诊断', '年龄','性别', '病程月']]
des_unmedicated_hc = scale_unmedicated_hc.describe()


#%% Extract covariances for all: age and sex
cov = pd.merge(uid, scale, left_on=0, right_on='folder', how='inner')[['folder','诊断', '年龄', '性别']]
cov['诊断'] = selected_diagnosis['诊断']
cov['性别'] = np.int32(cov['性别'] == 2)
cov.columns = ['folder', 'diagnosis', 'age', 'sex']
cov.to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\cov_550.txt', index=False)

#%% Extract covariances for unmedicated patients ans matched HC: age and sex
cov_unmedicated_sz_and_matched_hc = pd.merge(uid_unmedicated_sz_hc, cov, left_on=0, right_on='folder', how='inner')
cov_unmedicated_sz_and_matched_hc.drop(0, axis=1, inplace=True)
cov_unmedicated_sz_and_matched_hc.to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\cov_unmedicated_sp_and_hc_550.txt', index=False)
