"""
This script is a demo script, showing how to use eslearn to training and testing a SVC model.
"""
import numpy as np
import eslearn.machine_learning.classfication.pca_svc_cv as pca_svc

# =============================================================================
# All inputs
path_patients = 'root_path/patients/image files'  # .nii format
path_HC = 'root_path/controls/image_files'  # .nii format
path_mask = r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'  # mask file for filter image
path_out = r'D:\workstation_b\svc_results'  # directory for saving results
# =============================================================================

clf = pca_svc.PCASVCCV(
    path_patients=path_patients,
    path_HC=path_HC,
    path_mask=path_mask,
    path_out=path_out,
    num_of_kfold=5,  # How many folds to perform cross validation (Default: 5-fold cross validation)
    is_dim_reduction=1,  # Default is using PCA to reduce the dimension.
    components=0.75, 

    is_feature_selection=True,  # Whether perform feature selection( Default is using relief-based feature selection algorithms).
    n_features_to_select=0.8,  # How many features to be selected.
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