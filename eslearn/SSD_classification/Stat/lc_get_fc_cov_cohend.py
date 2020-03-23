# -*- coding: utf-8 -*-
"""
This script is used to do 3 things:
  1. Getting the functional connectivity networks for medicated SSD and first episode unmedicated SSD, as well as their matched HC.
  2. Extracting sorted covariance for medicated SSD and first episode unmedicated SSD, as well as their matched HC.
  3. Getting the Cohen'd values. 
"""
import sys
sys.path.append(r'D:\My_Codes\lc_rsfmri_tools_python\Workstation\SZ_classification\ML')
sys.path.append(r'D:\My_Codes\lc_rsfmri_tools_python\Statistics')
import numpy as np
import  pandas as pd
from lc_pca_svc_pooling import PCASVCPooling
import scipy.io as sio
from lc_calc_cohen_d_effective_size import CohenEffectSize

#%% Inputs
is_save = 1
dataset_first_episode_unmedicated_path = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_unmedicated_and_firstepisode_550.npy'
scale = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'
cov_550 = r'D:\WorkStation_2018\SZ_classification\Scale\cov_550.txt'
cov_206 = r'D:\WorkStation_2018\SZ_classification\Scale\cov_206.txt'
cov_COBRE = r'D:\WorkStation_2018\SZ_classification\Scale\cov_COBRE.txt'
cov_UCLA = r'D:\WorkStation_2018\SZ_classification\Scale\cov_UCLA.txt'
cov_unmedicated_sz_and_matched_hc = r'D:\WorkStation_2018\SZ_classification\Scale\cov_unmedicated_sp_and_hc_550.txt'

#%% Load all dataset
scale = pd.read_excel(scale)
sel = PCASVCPooling()
dataset_our_center_550 = np.load(sel.dataset_our_center_550)
dataset_206 = np.load(sel.dataset_206)
dataset_COBRE = np.load(sel.dataset_COBRE)
dataset_UCAL = np.load(sel.dataset_UCAL)
dataset1_firstepisodeunmed = np.load(dataset_first_episode_unmedicated_path)

cov_550, cov_206, cov_COBRE, cov_UCLA = pd.read_csv(cov_550), pd.read_csv(cov_206), pd.read_csv(cov_COBRE), pd.read_csv(cov_UCLA)
cov_all = pd.concat([cov_550, cov_206, cov_COBRE, cov_UCLA])
cov_feu = pd.read_csv(cov_unmedicated_sz_and_matched_hc)

# Extract ID
uid_our_center_550 = dataset_our_center_550[:, 0]
uid_206 = dataset_206[:, 0]
uid_COBRE = dataset_COBRE[:, 0]
uid_UCAL = dataset_UCAL[:, 0]
uid_feu = dataset1_firstepisodeunmed[:, 0]

# Extract features and label
features_our_center_550 = dataset_our_center_550[:, 2:]
features_206 = dataset_206[:, 2:]
features_COBRE = dataset_COBRE[:, 2:]
features_UCAL = dataset_UCAL[:, 2:]
fc_ssd_firstepisodeunmed = dataset1_firstepisodeunmed[:, 2:]

label_our_center_550 = dataset_our_center_550[:, 1]
label_206 = dataset_206[:, 1]
label_COBRE = dataset_COBRE[:, 1]
label_UCAL = dataset_UCAL[:, 1]
label_firstepisodeunmed = dataset1_firstepisodeunmed[:, 1]

#%% Get data and cov
# Medicated
uid_medicated_sp_hc_550 = np.int32(list(set(uid_our_center_550) - set(scale['folder'][(scale['诊断'] == 3) & (scale['用药'] != 1)])))
cov_medicated_sp_hc_550 = cov_550[cov_550['folder'].isin(uid_medicated_sp_hc_550)]
header = cov_medicated_sp_hc_550.columns
data_medicated_sp_hc_484 = pd.merge(pd.DataFrame(dataset_our_center_550), cov_medicated_sp_hc_550, left_on=0, right_on='folder', how='inner')
cov_medicated_sp_hc_484 = data_medicated_sp_hc_484[header]
features_our_center_484 = data_medicated_sp_hc_484.drop(header, axis=1)
features_our_center_484  = features_our_center_484.iloc[:, 2:]

cov_ssd_medicated = pd.DataFrame(np.concatenate([cov_medicated_sp_hc_550, cov_206, cov_COBRE, cov_UCLA], axis=0), columns=header)
# Add site id as covariance
cov_all_sites_id = pd.DataFrame(np.concatenate([np.ones([cov_medicated_sp_hc_550.shape[0],1]),
                          np.ones([cov_206.shape[0],1]) + 1,
                          np.ones([cov_COBRE.shape[0],1]) + 2,
                          np.ones([cov_UCLA.shape[0],1]) + 3])
    )

cov_ssd_medicated = pd.concat([cov_ssd_medicated['folder'], cov_all_sites_id, cov_ssd_medicated[['diagnosis', 'age', 'sex']]], axis=1)
                    
fc_ssd_medicated = np.concatenate([features_our_center_484, features_206, features_UCAL, features_COBRE], axis=0)
label_medicated = cov_ssd_medicated['diagnosis']

# First episode unmedicated
cov_feu = pd.merge(pd.DataFrame(uid_feu), cov_feu, left_on=0, right_on='folder', how='inner').drop(0, axis=1)

#%% Get the difference
# Medicated
fc_ssd_medicated = fc_ssd_medicated[label_medicated == 1]
data_hc_medicated = fc_ssd_medicated[label_medicated == 0]
cohen_medicated = CohenEffectSize(fc_ssd_medicated, data_hc_medicated)

# First episode unmedicated in dataset1
data_ssd_firstepisodeunmed = fc_ssd_firstepisodeunmed[label_firstepisodeunmed == 1]
data_hc_firstepisodeunmed = fc_ssd_firstepisodeunmed[label_firstepisodeunmed == 0]
cohen_feu = CohenEffectSize(data_ssd_firstepisodeunmed, data_hc_firstepisodeunmed)

#%% Make the differences to 2D matrix and save to mat
# All
cohen_medicated_full = np.zeros([246,246])
cohen_medicated_full[np.triu(np.ones([246,246]), 1) == 1] = cohen_medicated
cohen_medicated_full = cohen_medicated_full + cohen_medicated_full.T

cohen_feu_full = np.zeros([246,246])
cohen_feu_full[np.triu(np.ones([246,246]), 1) == 1] = cohen_feu
cohen_feu_full = cohen_feu_full + cohen_feu_full.T

#%% Save to mat for MATLAB process (NBS)
if is_save:
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\fc_medicated.mat', {'fc_medicated': fc_ssd_medicated})
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\fc_unmedicatedl.mat', {'fc_unmedicated': fc_ssd_firstepisodeunmed})
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\cov_ssd_medicated.mat', {'cov_ssd_medicated': cov_ssd_medicated.values})
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\cov_unmedicatedl.mat', {'cov_unmedicated': cov_feu.values})
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\Stat_results\cohen_medicated.mat', {'cohen_medicated': cohen_medicated_full})
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\Stat_results\cohen_feu.mat', {'cohen_feu': cohen_feu_full})