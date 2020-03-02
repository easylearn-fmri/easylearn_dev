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
path_patients = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_patient\Weighted'  # .nii format
path_HC = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\Weighted'  # .nii format
path_mask = r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'  # mask file for filter image
path_out = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree'  # directory for saving results
# =============================================================================

clf = pca_rfe_svc.PcaRfeSvcCV(
        path_patients=path_patients,
        path_HC=path_HC,
        path_mask=path_mask,
        path_out=path_out,
        data_preprocess_method='StandardScaler',
        data_preprocess_level='subject',
        num_of_fold_outer=5,  # How many folds to perform cross validation (Default: 5-fold cross validation)
        is_dim_reduction=1,  # Default is using PCA to reduce the dimension.
        components=0.95, 
        step=0.1,
        num_fold_of_inner_rfeCV=5,
        n_jobs=-1,
        is_showfig_finally=True,  # Whether show results figure finally.
        is_showfig_in_each_fold=False  # Whether show results in each fold.
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