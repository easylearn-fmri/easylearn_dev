# -*- coding: utf-8 -*-
"""
Created on 2019/11/20
All datasets were concatenate into one single dataset, then using cross-validation strategy.
This script is used to training a  linear svc model using a given training dataset, and validation this model using validation dataset.
Finally, we test the model using test dataset.
Dimension reduction: PCA
@author: LI Chao
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import numpy as np
import  pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import Utils.lc_niiProcessor as niiproc
import Utils.lc_dimreduction as dimreduction
from Utils.lc_evaluation import eval_performance


# =============================================================================
BD_path = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_patient\Weighted'
MDD_path = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\Weighted'
HC_path = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\Weighted'
# =============================================================================


class PCASVCPooling():
    def __init__(sel,
                 dataset1_path = BD_path,
                 dataset2_path = MDD_path,
                 dataset3_path = HC_path,

                 is_dim_reduction=0,
                 components=0.80,
                 numofcv=3,
                 show_results=1,
                 show_roc=1):

        sel.dataset1_path = dataset1_path
        sel.dataset2_path = dataset2_path
        sel.dataset3_path = dataset3_path

        sel.is_dim_reduction = is_dim_reduction
        sel.components = components
        sel.numofcv = numofcv
        sel.show_results = show_results
        sel.show_roc = show_roc

    def main_function(sel):
        """
        The training data, validation data and  test data are randomly splited
        """
        print('training model and testing...\n')

        # load data
        dataset1 = sel.loadnii(sel.dataset1_path, '.nii')
        dataset2 = sel.loadnii(sel.dataset2_path, '.nii')
        dataset3 = sel.loadnii(sel.dataset3_path, '.nii')
        data_all = np.vstack([dataset1,dataset2,dataset3])
        label_all = np.hstack([np.ones([len(dataset1),])-1,np.ones([len(dataset2),]),np.ones([len(dataset2),])+1])

        # KFold Cross Validation
        label_test_all = np.array([], dtype=np.int16)
        train_index = np.array([], dtype=np.int16)
        test_index = np.array([], dtype=np.int16)
        sel.decision = np.array([], dtype=np.int16)
        sel.prediction = np.array([], dtype=np.int16)
        sel.accuracy = np.array([], dtype=np.float16)
        sel.sensitivity = np.array([], dtype=np.float16)
        sel.specificity = np.array([], dtype=np.float16)
        sel.AUC = np.array([], dtype=np.float16)
        kf = KFold(n_splits=sel.numofcv, shuffle=True, random_state=0)
        for i, (tr_ind, te_ind) in enumerate(kf.split(data_all)):
            print(f'------{i+1}/{sel.numofcv}...------\n')
            train_index = np.int16(np.append(train_index, tr_ind))
            test_index = np.int16(np.append(test_index, te_ind))
            feature_train = data_all[tr_ind, :]
            label_train = label_all[tr_ind]
            feature_test = data_all[te_ind, :]
            label_test = label_all[te_ind]
            label_test_all = np.int16(np.append(label_test_all, label_test))

            # resampling training data
            feature_train, label_train = sel.re_sampling(
                feature_train, label_train)

            # normalization
            feature_train = sel.normalization(feature_train)
            feature_test = sel.normalization(feature_test)

            # dimension reduction using univariate feature selection
            feature_train, feature_test, mask_selected = sel.dimReduction_filter(
                    feature_train, label_train, feature_test, 0.01)

            # dimension reduction
            if sel.is_dim_reduction:
                feature_train, feature_test, model_dim_reduction = sel.dimReduction(
                    feature_train, feature_test, sel.components)
                print(f'After dimension reduction, the feature number is {feature_train.shape[1]}')
            else:
                print('No dimension reduction perfromed\n')

            # train and test
            print('training and testing...\n')
            model = sel.training(feature_train, label_train)
            if sel.is_dim_reduction:
                sel.coef = model_dim_reduction.inverse_transform(model.coef_)
            else:
                sel.coef = model.coef_

            pred, dec = sel.testing(model, feature_test)
            sel.prediction = np.append(sel.prediction, np.array(pred))
            sel.decision = np.append(sel.decision, np.array(dec))

            # Evaluating classification performances
            acc, sens, spec, auc = eval_performance(label_test, pred, dec, sel.show_roc)
                
            sel.accuracy = np.append(sel.accuracy, acc)
            sel.sensitivity = np.append(sel.sensitivity, sens)
            sel.specificity = np.append(sel.specificity, spec)
            sel.AUC = np.append(sel.AUC, auc)
            print(f'performances = {acc, sens, spec,auc}')
        print('Done!')
        return sel

    def loadnii(sel, data_path, suffix):
        niip = niiproc.NiiProcessor()
        data, _ = niip.main(data_path, suffix)
        data = np.squeeze(np.array([np.array(data).reshape(1, -1) for data in data]))
        return data

    def re_sampling(sel, feature, label):
        """
        Used to over-sampling unbalanced data
        """
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        feature_resampled, label_resampled = ros.fit_resample(feature, label)
        from collections import Counter
        print(sorted(Counter(label_resampled).items()))
        return feature_resampled, label_resampled

    def normalization(sel, data):
        '''
        Because of our normalization level is on subject, 
        we should transpose the data matrix on python(but not on matlab)
        '''
        scaler = preprocessing.StandardScaler().fit(data.T)
        z_data = scaler.transform(data.T) .T
        return z_data
    
    def dimReduction_filter(sel, feature_train, label_train, feature_test, p_thrd = 0.05):
        """
        This function is used to Univariate Feature Selection:: ANOVA
        """
        from sklearn.feature_selection import f_classif
        f, p = f_classif(feature_train, label_train)
        mask_selected = p < p_thrd
        feature_train = feature_train[:,mask_selected]
        feature_test = feature_test[:, mask_selected]
        return feature_train, feature_test, mask_selected
        
    def dimReduction(self, train_X, test_X, pca_n_component):
        F_statistic, pVal = stats.f_oneway(group1, group2, group3)
        train_X, trained_pca = dimreduction.pca(train_X, pca_n_component)
        test_X = trained_pca.transform(test_X)
        return train_X, test_X, trained_pca

    def training(sel, train_X, train_y):
        # svm GrigCV
        svc = svm.SVC(kernel='linear', C=1, class_weight='balanced', max_iter=5000, random_state=0)
        svc.fit(train_X, train_y)
        return svc

    def testing(sel, model, test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict, decision


#
if __name__ == '__main__':
    sel = PCASVCPooling()
    results = sel.main_function()
    results = results.__dict__
    
    print(np.mean(results['accuracy']))
    print(np.std(results['accuracy']))

    print(np.mean(results['sensitivity']))
    print(np.std(results['sensitivity']))

    print(np.mean(results['specificity']))
    print(np.std(results['specificity']))

