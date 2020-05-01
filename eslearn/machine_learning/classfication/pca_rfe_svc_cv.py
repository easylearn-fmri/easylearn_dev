# -*- coding: utf-8 -*-
"""
Created on 2020/02/08
Classifier: linear SVC
Dimension reduction: PCA
Feature selection: RFE
------
@author: LI Chao
"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import preprocessing
import nibabel as nib
import os

from eslearn.utils.lc_evaluation_model_performances import eval_performance
from eslearn.utils.lc_niiProcessor import NiiProcessor
from eslearn.feature_engineering.feature_selection.el_rfe import rfeCV
import eslearn.feature_engineering.feature_preprocessing.el_preprocessing as elprep
from eslearn.feature_engineering.feature_reduction.el_dimreduction import pca_apply


class PcaRfeSvcCV():
    """
    This class is used to execute pca-rfe-svc-based classification training and testing.
    Classifier is Linear SVC.
    Dimension reduction method: PCA.
    Feature selection: RFECV.
    NOTE: Input data must be in the .nii or .img format.

    Parameters
    ----------
    path_patients :    
        Path of the image files of patients

    path_HC : 
        Path of the image files of HC 

    path_mask : 
        Path of the mask

    path_out : 
        Path to save results

    data_preprocess_method: str
        How to preprocess features 'StandardScaler' OR 'MinMaxScaler'.

    data_preprocess_level: str
        Which level to preprocess features. 'group' or 'subject'
        
    is_dim_reduction : bool
        If perfrome dimension reduction.

    n_components: float from 0 to 1
        If is_dim_reduction, then how many components to remain.

    step: float (0, n_features)
        RFE step, eliminate how many features each iteration,
        If float is (0, 1), then eliminate step * n_features features each iteration.
        If float is [1, n_features], then eliminate step features each iteration.

    num_fold_of_inner_rfeCV: int 
        How many folds of inner RFECV.

    n_jobs: int
        How many parallel jobs.

    num_of_fold_outer: int
        Number of the k in k-fold cross-validation

    is_showfig_finally: bool
        If show figure after all iteration finished.

    is_showfig_in_each_fold: bool
        If show figure in each fold.

    Returns
    -------
    Save all classification results and figures to local disk.
    """
    def __init__(self,
                 path_patients=None,
                 path_HC=None,
                 path_mask=None,
                 path_out=None,
                 data_preprocess_method='MinMaxScaler',
                 data_preprocess_level='subject',
                 num_of_fold_outer=5,
                 is_dim_reduction=1,
                 components=0.95,
                 step=0.1,
                 num_fold_of_inner_rfeCV=5,
                 n_jobs=-1,
                 is_showfig_finally=True,
                 is_showfig_in_each_fold=False):

         self.path_patients = path_patients
         self.path_HC = path_HC
         self.path_mask = path_mask
         self.path_out = path_out
         self.data_preprocess_method = data_preprocess_method
         self.data_preprocess_level = data_preprocess_level
         self.num_of_fold_outer =  num_of_fold_outer
         self.is_dim_reduction = is_dim_reduction
         self.components = components
         self.step = step
         self.num_fold_of_inner_rfeCV = num_fold_of_inner_rfeCV
         self.n_jobs = n_jobs
         self.is_showfig_finally = is_showfig_finally
         self.is_showfig_in_each_fold = is_showfig_in_each_fold


    def main_function(self):
        """
        This function is the main function.
        """

        # Load data and mask
        data_all, label_all, self.orig_shape, self.mask_obj, self.mask_all = self._load_nii_and_gen_label()

        # KFold Cross Validation
        self.label_test_all = np.array([], dtype=np.int16)
        train_index = np.array([], dtype=np.int16)
        test_index = np.array([], dtype=np.int16)
        self.decision = np.array([], dtype=np.int16)
        self.prediction = np.array([], dtype=np.int16)
        self.accuracy = np.array([], dtype=np.float16)
        self.sensitivity = np.array([], dtype=np.float16)
        self.specificity = np.array([], dtype=np.float16)
        self.AUC = np.array([], dtype=np.float16)
        self.coef = []
        kf = KFold(n_splits=self.num_of_fold_outer, shuffle=True, random_state=0)
        for i, (tr_ind, te_ind) in enumerate(kf.split(data_all)):
            print(f'------{i+1}/{self.num_of_fold_outer}...------\n')
            train_index = np.int16(np.append(train_index, tr_ind))
            test_index = np.int16(np.append(test_index, te_ind))
            feature_train = data_all[tr_ind, :]
            label_train = label_all[tr_ind]
            feature_test = data_all[te_ind, :]
            label_test = label_all[te_ind]
            self.label_test_all = np.int16(np.append(self.label_test_all, label_test))

            # Resampling training data
            feature_train, label_train = self.re_sampling(feature_train, label_train)

            # data_preprocess
            prep = elprep.Preprocessing(self.data_preprocess_method, self.data_preprocess_level)
            feature_train, feature_test = prep.data_preprocess(feature_train, feature_test)

            # dimension reduction using univariate feature selection
            # feature_train, feature_test, mask_selected = self.dimReduction_filter(
            #         feature_train, label_train, feature_test, 0.05)

            # Dimension reduction using PCA
            if self.is_dim_reduction:
                feature_train, feature_test, model_dim_reduction = self.dimReduction_PCA(
                    feature_train, feature_test, self.components)
                print(f'After dimension reduction, the feature number is {feature_train.shape[1]}')
            else:
                print('No dimension reduction perfromed\n')
                print(f'The feature number is {feature_train.shape[1]}')
                
            # Train: inner feature selection using RFECV
            print('Training...\n')
            model, weight = self.rfeCV_training(feature_train, label_train, self.step, self.num_fold_of_inner_rfeCV, self.n_jobs)
                
            if self.is_dim_reduction:
                self.coef.append(model_dim_reduction.inverse_transform(weight))
            else:
                self.coef.append(weight)
            
            # Testting
            print('Testting...\n')
            pred, dec = self.testing(model, feature_test)
            self.prediction = np.append(self.prediction, np.array(pred))
            self.decision = np.append(self.decision, np.array(dec))

            # Evaluating classification performances
            acc, sens, spec, auc = eval_performance(label_test, pred, dec, 
                accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                 verbose=1, is_showfig=self.is_showfig_in_each_fold)

            self.accuracy = np.append(self.accuracy, acc)
            self.sensitivity = np.append(self.sensitivity, sens)
            self.specificity = np.append(self.specificity, spec)
            self.AUC = np.append(self.AUC, auc)
            
        # Save results and fig to local path
        self.save_results()
        self._weight2nii(dimension_nii_data=(61, 73, 61))
        self.save_fig()
            
        print("--" * 10 + "Done!" + "--" * 10 )
        return self


    def _load_nii_and_gen_label(self):
            """
            Load nii and generate label
            """
            data1, _ = NiiProcessor().read_multi_nii(self.path_patients)
            data1 = np.squeeze(
                np.array([np.array(data1).reshape(1, -1) for data1 in data1]))
    
            data2, _ = NiiProcessor().read_multi_nii(self.path_HC)
            data2 = np.squeeze(
                np.array([np.array(data2).reshape(1, -1) for data2 in data2]))
    
            data = np.vstack([data1, data2])
    
            # data in mask
            mask, mask_obj = NiiProcessor().read_sigle_nii(self.path_mask)
            orig_shape = mask.shape
            mask = mask >= 0.2
            mask = np.array(mask).reshape(-1,)

            data_in_mask = data[:, mask]
            # label
            label = np.hstack(
                [np.ones([len(data1), ]), np.ones([len(data2), ])-1])
            return data_in_mask, label, orig_shape, mask_obj, mask
    

    def re_sampling(self, feature, label):
        """
        Used to over-sampling unbalanced data
        """
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        feature_resampled, label_resampled = ros.fit_resample(feature, label)
        from collections import Counter
        print(f"After re-sampling, the sample size are: {sorted(Counter(label_resampled).items())}")
        return feature_resampled, label_resampled

    def data_preprocess(sel, feature_train, feature_test, data_preprocess_method, data_preprocess_level):
        '''
        This function is used to preprocess features
        Method 1: preprocess data in group level, one feature by one feature.
        Method 2: preprocess data in subject level.
        '''
        # Method 1: Group level preprocessing.
        if data_preprocess_level == 'group':
            feature_train, model = elscaler.scaler(feature_train, data_preprocess_method)
            feature_test = model.transform(feature_test)
        elif data_preprocess_level == 'subject':
            # Method 2: Subject level preprocessing.
            scaler = preprocessing.StandardScaler().fit(feature_train.T)
            feature_train = scaler.transform(feature_train.T) .T
            scaler = preprocessing.StandardScaler().fit(feature_test.T)
            feature_test = scaler.transform(feature_test.T) .T
        else:
            print('Please provide which level to preprocess features\n')
            return

        return feature_train, feature_test
    
    def feature_selection_relief(self, feature_train, label_train, feature_test, n_features_to_select=None):
        """
        This functio is used to select the features using relief-based feature selection algorithms
        """
        from skrebate import ReliefF
        
        [n_sub, n_features] = np.shape(feature_train)
        if n_features_to_select is None: 
            n_features_to_select = np.int(np.round(n_features / 10))
            
        if isinstance(n_features_to_select, np.float): 
            n_features_to_select = np.int(np.round(n_features * n_features_to_select))
        
        fs = ReliefF(n_features_to_select=n_features_to_select, 
                     n_neighbors=100, discrete_threshold=10, verbose=True, n_jobs=-1)
        fs.fit(feature_train, label_train)
        feature_train = fs.transform(feature_train)
        feature_test = fs.transform(feature_test)
        mask = fs.top_features_[:n_features_to_select]
        return feature_train, feature_test, mask, n_features
        
        
    def dimReduction_filter(self, feature_train, label_train, feature_test, p_thrd = 0.05):
        """
        This function is used to Univariate Feature Selection:: ANOVA
        """
        from sklearn.feature_selection import f_classif
        f, p = f_classif(feature_train, label_train)
        mask_selected = p < p_thrd
        feature_train = feature_train[:,mask_selected]
        feature_test = feature_test[:, mask_selected]
        return feature_train, feature_test, mask_selected
        
    def dimReduction_PCA(self, train_X, test_X, pca_n_component):
        x_train, x_test, trained_pca = pca_apply(
                    train_X, test_X, pca_n_component)
        return x_train, x_test, trained_pca

    def training(self, train_X, train_y):
        # Classfier is SVC
        svc = svm.SVC(kernel='linear', C=1, class_weight='balanced', max_iter=5000, random_state=0)
        svc.fit(train_X, train_y)
        return svc

    def rfeCV_training(self, train_X, train_y, step, num_fold_of_inner_rfeCV, n_jobs):
        model, weight = rfeCV(train_X, train_y, step, num_fold_of_inner_rfeCV, n_jobs)
        return model, weight

    def testing(self, model, test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict, decision

    def save_results(self):
        # Save performances and others
        import pandas as pd
        performances_to_save = np.concatenate([[self.accuracy], [self.sensitivity], [self.specificity], [self.AUC]], axis=0).T
        de_pred_label_to_save = np.concatenate([[self.decision], [self.prediction], [self.label_test_all]], axis=0).T

        performances_to_save = pd.DataFrame(performances_to_save, columns=[['Accuracy','Sensitivity', 'Specificity', 'AUC']])
        de_pred_label_to_save = pd.DataFrame(de_pred_label_to_save, columns=[['Decision','Prediction', 'Sorted_Real_Label']])
        
        performances_to_save.to_csv(os.path.join(self.path_out, 'Performances.txt'), index=False, header=True)
        de_pred_label_to_save.to_csv(os.path.join(self.path_out, 'Decision_prediction_label.txt'), index=False, header=True)

        
    def _weight2nii(self, dimension_nii_data=(61, 73, 61)):
        """
        Transfer weight matrix to nii file
        I used the mask file as reference to generate the nii file
        """
        weight = np.squeeze(self.coef)
        weight_mean = np.mean(weight, axis=0)

        # to orignal space
        weight_mean_orig = np.zeros(np.size(self.mask_all))
        weight_mean_orig[self.mask_all] = weight_mean
        weight_mean_orig =  np.reshape(weight_mean_orig, dimension_nii_data)
        # save to nii
        weight_nii = nib.Nifti1Image(weight_mean_orig, affine=self.mask_obj.affine)
        weight_nii.to_filename(os.path.join(self.path_out, 'weight.nii'))
        
    def save_fig(self):
        # Save ROC and Classification 2D figure
        acc, sens, spec, auc = eval_performance(self.label_test_all, self.prediction, self.decision, 
                                                self.accuracy, self.sensitivity, self.specificity, self.AUC,
                                                verbose=0, is_showfig=self.is_showfig_finally, is_savefig=1, 
                                                out_name=os.path.join(self.path_out, 'Classification_performances.pdf'))

#
if __name__ == '__main__':
    # =============================================================================
    # All inputs
    path_patients = r'D:\workstation_b\xiaowei\ToLC\training\BD_label1'  # 训练组病人
    path_HC = r'D:\workstation_b\xiaowei\ToLC\training\MDD__label0' # 训练组正常人
    path_mask = r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'
    path_out = r'D:\workstation_b\xiaowei\ToLC'
    # =============================================================================
    
    clf = PcaRfeSvcCV(path_patients=path_patients,
                        path_HC=path_HC,
                        path_mask=path_mask,
                        path_out=path_out,
                        is_dim_reduction=1,
                        components=0.95)

    clf.main_function()

    print(clf.accuracy)
    
    print(f"mean accuracy = {np.mean(clf.accuracy)}")
    print(f"std of accuracy = {np.std(clf.accuracy)}")

    print(f"mean sensitivity = {np.mean(clf.sensitivity)}")
    print(f"std of sensitivity = {np.std(clf.sensitivity)}")

    print(f"mean specificity = {np.mean(clf.specificity)}")
    print(f"std of specificity = {np.std(clf.specificity)}")

    print(f"mean AUC = {np.mean(clf.AUC)}")
    print(f"std of AUC = {np.std(clf.AUC)}")
