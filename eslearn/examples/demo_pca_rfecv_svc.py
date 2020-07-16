"""
This script is a demo script, showing how to use eslearn to training and testing a SVC model.
Classifier: linear SVC
Dimension reduction: PCA
Feature selection: RFE
"""

import numpy as np
import eslearn.machine_learning.classfication.pca_rfe_svc_cv as pca_rfe_svc

# =============================================================================
# All inputs
dataset_patients = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_patient\Weighted'  # All patients' image files, .nii format
dataset_HC = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\Weighted'  # All HCs' image files, .nii format
mask = r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'  # Mask file for filter image
outpath = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree'  # Directory for saving results
data_preprocess_method='StandardScaler'
data_preprocess_level='group'  # In which level to preprocess data 'subject' or 'group'
num_of_fold_outer = 5  # How many folds to perform cross validation
is_dim_reduction = 1  # Whether to perform dimension reduction, default is using PCA to reduce the dimension.
components = 0.95   # How many percentages of the cumulatively explained variance to be retained. This is used to select the top principal components.
step = 0.1  # RFE parameter: percentages or number of features removed each iteration.
num_fold_of_inner_rfeCV = 5  # RFE parameter:  how many folds to perform inner RFE loop.
n_jobs = -1  # RFE parameter:  how many jobs (parallel works) to perform inner RFE loop.
is_showfig_finally = True  # Whether show results figure finally.
is_showfig_in_each_fold = False  # Whether show results in each fold.
# =============================================================================

clf = pca_rfe_svc.PcaRfeSvcCV(
        dataset_patients=dataset_patients,
        dataset_HC=dataset_HC,
        mask=mask,
        outpath=outpath,
        data_preprocess_method=data_preprocess_method,
        data_preprocess_level=data_preprocess_level,
        num_of_fold_outer=num_of_fold_outer, 
        is_dim_reduction=is_dim_reduction,  
        components=components, 
        step=step,
        num_fold_of_inner_rfeCV=num_fold_of_inner_rfeCV,
        n_jobs=n_jobs,
        is_showfig_finally=is_showfig_finally,  
        is_showfig_in_each_fold=is_showfig_in_each_fold  
    )

results = clf.main_function()
results = results.__dict__

print(f"mean accuracy = {np.mean(results['accuracy'])}")
print(f"std of accuracy = {np.std(results['accuracy'])}")

print(f"mean sensitivity = {np.mean(results['sensitivity'])}")
print(f"std of sensitivity = {np.std(results['sensitivity'])}")

print(f"mean specificity = {np.mean(results['specificity'])}")
print(f"std of specificity = {np.std(results['specificity'])}")

print(f"mean AUC = {np.mean(results['AUC'])}")
print(f"std of AUC = {np.std(results['AUC'])}")