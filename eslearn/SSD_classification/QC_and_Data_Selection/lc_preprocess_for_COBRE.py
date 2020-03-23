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
matroot = r'D:\WorkStation_2018\SZ_classification\Data\SelectedFC_COBRE'  # all mat files directory
scale = r'H:\Data\精神分裂症\COBRE\COBRE_phenotypic_data.csv'  # whole scale path
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
allsubjname = pd.DataFrame([name[0] for name in allsubjname])
scale_data = pd.read_csv(scale,sep=',',dtype='str')
print(scale_data)
diagnosis = pd.merge(allsubjname,scale_data,left_on=0,right_on='ID')[['ID','Subject Type']]
scale_data = pd.merge(allsubjname,scale_data,left_on=0,right_on='ID')

diagnosis['Subject Type'][diagnosis['Subject Type'] == 'Control'] = 0
diagnosis['Subject Type'][diagnosis['Subject Type'] == 'Patient'] = 1
include_loc = diagnosis['Subject Type'] != 'Disenrolled'
diagnosis = diagnosis[include_loc.values]
allmat = allmat[include_loc.values]
allsubjname = allsubjname[include_loc.values]

diagnosis = np.array(np.int32(diagnosis))

allmat_plus_label = np.concatenate([diagnosis, allmat], axis=1)
print(allmat_plus_label.shape)
# np.save(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\COBRE.npy',allmat_plus_label)

#%% Extract covariances: age and sex
cov = pd.merge(allsubjname,scale_data,left_on=0,right_on='ID')[['ID','Subject Type', 'Current Age', 'Gender']]
cov[['ID','Subject Type']] = diagnosis
cov['Gender'] = cov['Gender'] == 'Male'
cov = pd.DataFrame(np.int64(cov))
cov.columns = ['folder', 'diagnosis', 'age', 'sex']
cov.to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\cov_COBRE.txt', index=False)
