"""
This script is used to transform the UCLA dataset into .npy format.
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
matroot = r'D:\WorkStation_2018\SZ_classification\Data\SelectedFC_UCLA'
scale = r'H:\Data\精神分裂症\ds000030\schizophrenia_UCLA_restfmri\participants.tsv'
n_node = 246  #  number of nodes in the mat network

# Transform the .mat files to one .npy file
allmatname = os.listdir(matroot)
allmatpath = [os.path.join(matroot, matpath) for matpath in allmatname]
mask = np.triu(np.ones(n_node),1)==1
allmat = [read_mat(matpath)[mask].T for matpath in allmatpath]
allmat = np.array(allmat,dtype=np.float32)


# Give labels to each subject, concatenate at the first column
allmatname = pd.DataFrame(allmatname)
allsubjname = allmatname.iloc[:,0].str.findall(r'[1-9]\d*')
allsubjname = pd.DataFrame(['sub-' + name[0] for name in allsubjname])
scale_data = pd.read_csv(scale,sep='\t')
diagnosis = pd.merge(allsubjname,scale_data,left_on=0,right_on='participant_id')[['participant_id','diagnosis']]
diagnosis['diagnosis'][diagnosis['diagnosis'] == 'CONTROL']=0
diagnosis['diagnosis'][diagnosis['diagnosis'] == 'SCHZ']=1
diagnosis['participant_id'] = diagnosis['participant_id'].str.replace('sub-', '')

label = np.array(np.int32(diagnosis))
allmat_plus_label = np.concatenate([label, allmat], axis=1)
print(allmat_plus_label.shape)
# np.save(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\UCLA.npy',allmat_plus_label)
#
# d1=np.load(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\UCLA.npy')
# d2=np.load(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\UCLA_rest.npy')
# d = np.concatenate([d1,d2],axis=0)
# np.save(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\UCLA_all.npy',d)
# print(d.shape)

#%% Extract covariances: age and sex
cov = pd.merge(allsubjname,scale_data,left_on=0,right_on='participant_id')[['participant_id','diagnosis', 'age', 'gender']]
cov[['participant_id', 'diagnosis']] = diagnosis[['participant_id', 'diagnosis']] 
cov['gender'] = cov['gender'] == 'M'
cov = pd.DataFrame(np.int64(cov))
cov.columns = ['folder', 'diagnosis', 'age', 'sex']
cov.to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\cov_UCLA.txt', index=False)
