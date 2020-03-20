# -*- coding: utf-8 -*-
"""
This script is used to get the average fc for SZ and HC.
NOTO. This part of script is only used to preprocess the data, 
then submit the data to MATLAB to add weight mask and visualization.
"""

import sys
sys.path.append(r'D:\My_Codes\lc_rsfmri_tools_python\Workstation\SZ_classification\ML')
sys.path.append(r'D:\My_Codes\lc_rsfmri_tools_python\Statistics')
import numpy as np
import  pandas as pd
from lc_pca_svc_pooling import PCASVCPooling
import scipy.io as sio
from scipy.stats import ttest_ind
from sklearn.feature_selection import SelectFdr
from mne.stats import fdr_correction, bonferroni_correction


is_save = 1
# Unique index of first episode unmedicated patients
dataset_first_episode_unmedicated_path = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_unmedicated_and_firstepisode_550.npy'
scale = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'

#%% Load all dataset
scale = pd.read_excel(scale)
sel = PCASVCPooling()
dataset_our_center_550 = np.load(sel.dataset_our_center_550)
dataset_206 = np.load(sel.dataset_206)
dataset_COBRE = np.load(sel.dataset_COBRE)
dataset_UCAL = np.load(sel.dataset_UCAL)
dataset1_firstepisodeunmed = np.load(dataset_first_episode_unmedicated_path)

# Extract features and label
features_our_center_550 = dataset_our_center_550[:, 2:]
features_206 = dataset_206[:, 2:]
features_COBRE = dataset_COBRE[:, 2:]
features_UCAL = dataset_UCAL[:, 2:]
feature_firstepisodeunmed = dataset1_firstepisodeunmed[:, 2:]

label_our_center_550 = dataset_our_center_550[:, 1]
label_206 = dataset_206[:, 1]
label_COBRE = dataset_COBRE[:, 1]
label_UCAL = dataset_UCAL[:, 1]
label_firstepisodeunmed = dataset1_firstepisodeunmed[:, 1]

#%% Generate training data and test data
# All
data_all = np.concatenate(
    [features_our_center_550, features_206, features_UCAL, features_COBRE], axis=0)
label_all = np.concatenate(
    [label_our_center_550, label_206, label_UCAL, label_COBRE], axis=0)


#%% Get the difference
FD = SelectFdr(ttest_ind, 0.05)  # FDR correction:Benjamini-Hochberg procedure
# All datasets
data_sz = data_all[label_all == 1]
average_fc_sz_all = np.mean(data_sz, axis=0)

data_sz_firstepisodeunmed = feature_firstepisodeunmed[label_firstepisodeunmed == 1]
average_fc_sz = np.mean(data_sz_firstepisodeunmed,axis=0)

np.corrcoef(average_fc_sz, average_fc_sz_all)
#%% Make the differences to 2D matrix and save to mat
average_fc_all = np.zeros([246,246])
average_fc_all[np.triu(np.ones([246,246]), 1) == 1] = average_fc_sz_all

average_fc_unmedicated = np.zeros([246,246])
average_fc_unmedicated[np.triu(np.ones([246,246]), 1) == 1] = average_fc_sz

#%% save
if is_save:
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Figure\average_fc_all.mat', {'average_fc_all': average_fc_all})
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Figure\average_fc_unmedicated.mat', {'average_fc_unmedicated': average_fc_unmedicated})

