# -*- coding: utf-8 -*-
"""
Created on 2020/03/16
------
@author: LI Chao
"""

import os
import numpy as np
import pandas as pd
import xlwt
from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import LassoCV, Lasso
from sklearn.externals import joblib
from sklearn.linear_model import lasso_path, enet_path
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn import  preprocessing

from eslearn.utils.lc_evaluation_model_performances import eval_performance
from eslearn.utils.el_preprocessing import Preprocessing
from eslearn.utils.lc_niiProcessor import NiiProcessor
import eslearn.utils.el_preprocessing as elprep


class ClassifyFourKindOfPersonTrain():
    """
    This class is used to training and validating classification model for 2 kind of sensitive person identification.

    Parameters
    ----------
    data_train_file: path str 
        Path of the training dataset

    data_validation_file: path str 
        Path of the test dataset

    label_train_file: path str 
        Path of the training label

    label_validation_file: path str 
        Path of the test label

    path_out : 
        Path to save results
       
    is_feature_selection : bool
        if perfrome feature selection.

    is_showfig_finally: bool
        If show figure after all iteration finished.

    Returns
    -------
    Save all classification results and figures to local disk.
    """
    def __init__(selftest,
                 data_train_file=None,
                 data_validation_file=None,
                 label_train_file=None,
                 label_validation_file=None,
                 path_out=None,
                 is_feature_selection=False,
                 n_features_to_select=None,
                 is_showfig_finally=True,
                 rand_seed=666):

         selftest.data_train_file = data_train_file
         selftest.data_validation_file = data_validation_file
         selftest.label_train_file = label_train_file
         selftest.label_validation_file = label_validation_file
         selftest.path_out = path_out
         selftest.n_features_to_select = n_features_to_select
         selftest.is_feature_selection = is_feature_selection
         selftest.is_showfig_finally = is_showfig_finally
         selftest.rand_seed = rand_seed


    def main_function(selftest):
        """
        """
        print('Training model and testing...\n')

        # load data
        feature_train, feature_validation, label_train, label_validation, colname = selftest._load_data()
        n_features_orig = feature_train.shape[1]
        
        # Check data
        
        # Age encoding
        feature_train[:, 2] = selftest.age_encodeing(feature_train[:,2], feature_train[:,2])
        feature_validation[:, 2] = selftest.age_encodeing(feature_train[:,2], feature_validation[:,2])

        # Data normalization: do not need, because all variables are discrete variables.
        
        # Feature selection: LassoCV
        if selftest.is_feature_selection:       
            coef, mask_lassocv = selftest.feature_selection_lasso(feature_train, label_train)
            feature_train, feature_validation = feature_train[:, mask_lassocv], feature_validation[:, mask_lassocv]  
            var_important = pd.DataFrame(np.array(colname)[mask_lassocv])
            var_important_coef = pd.concat([var_important, pd.DataFrame(coef[coef != 0])], axis=1)
            var_important_coef.columns=['变量', '系数(lasso); 正系数为危险因素，负系数为保护因素']
            var_important_coef.to_csv(os.path.join(selftest.path_out, 'important_variables.txt'), index=False)

                
        # Onehot encoding
        # onehot = OneHotEncoder()
        # onehot.fit(feature_train)
        # feature_train = onehot.transform(feature_train).toarray()
        # feature_validation= onehot.transform(feature_validation).toarray()
        
        # Train
        print('training and testing...\n')
        if selftest.is_feature_selection:
            model = selftest.training(feature_train, label_train)
        else:
            model, w = selftest.rfeCV(feature_train, label_train)

        # Save model
        with open(os.path.join(selftest.path_out, 'model_classification.pkl'), 'wb') as f_model:
            joblib.dump(model, f_model)

        # Validating
        prediction_train, decision_train = selftest.testing(model, feature_train)
        prediction_validation, decision_validation = selftest.testing(model, feature_validation)
        
        # Evaluating classification performances
        accuracy_train, sensitivity_train, specificity_train, AUC_train = eval_performance(label_train, prediction_train, decision_train, 
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
             verbose=1, is_showfig=0)

        accuracy_validation, sensitivity_validation, specificity_validation, AUC_validation = eval_performance(label_validation,prediction_validation, decision_validation, 
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
              verbose=1, is_showfig=0)

        # Save results and fig to local path
        selftest.save_results(accuracy_train, sensitivity_train, specificity_train, AUC_train, 
            decision_train, prediction_train, label_train, 'train')

        selftest.save_results(accuracy_validation, sensitivity_validation, specificity_validation, AUC_validation, 
            decision_validation, prediction_validation, label_validation, 'validation')

        selftest.save_fig(label_train, prediction_train, decision_train, accuracy_train, 
            sensitivity_train, specificity_train, AUC_train, 
            'classification_performances_train.pdf')

        selftest.save_fig(label_validation, prediction_validation, decision_validation, accuracy_validation, 
            sensitivity_validation, specificity_validation, AUC_validation, 
            'classification_performances_validation.pdf')
        
        print(f"MSE = {np.mean(np.power((decision_validation - label_validation), 2))}")
            
        print("--" * 10 + "Done!" + "--" * 10 )
        return selftest


    def _load_data(selftest):
        """
        Load data
        """
        data_all_file = r'D:\workstation_b\Fundation\给黎超.xlsx'
        data_all = pd.read_excel(data_all_file)
        colname = np.array(data_all.columns)[np.arange(2,18)]
        data_train = np.load(selftest.data_train_file)
        data_validation = np.load(selftest.data_validation_file)
        label_train = np.load(selftest.label_train_file)
        label_validation = np.load(selftest.label_validation_file)
        return data_train, data_validation, label_train, label_validation, colname
    
    def feature_selection_lasso(selftest, feature, label):
        lc = LassoCV(cv=10, alphas=np.linspace(pow(10, -2), pow(10, 1), 1000))
        lc.fit(feature, label)
        with open(os.path.join(selftest.path_out, 'mask_selected_features_lassocv.pkl'), 'wb') as f_mask:
            joblib.dump(lc.coef_, f_mask)

        return lc.coef_, lc.coef_ != 0
        
        
    def feature_selection_relief(selftest, feature_train, label_train, feature_validation, n_features_to_select=None):
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
        feature_validation = fs.transform(feature_validation)
        mask = fs.top_features_[:n_features_to_select]
        return feature_train, feature_validation, mask, n_features, fs
    
    def age_encodeing(selftest, age_train, age_target):
        """
        Encoding age_target to separate variable
        """
        sep = pd.DataFrame(age_train).describe()
        age_target[age_target < sep.loc['25%'].values] = 0
        age_target[(age_target >= sep.loc['25%'].values) & (age_target < sep.loc['50%'].values)] = 1
        age_target[(age_target >= sep.loc['50%'].values) & (age_target < sep.loc['75%'].values)] = 2
        age_target[age_target >= sep.loc['75%'].values] = 3 
        return age_target
    
    def rfeCV(selftest, train_x, train_y, step=1, cv=10, n_jobs=-1, permutation=0):
        """
        Nested rfe
        """
        n_samples, n_features = train_x.shape
        estimator = SVC(kernel="linear")
        model = RFECV(estimator, step=step, cv=cv, n_jobs=n_jobs)
        model = model.fit(train_x, train_y)
        mask = model.support_
        optmized_model = model.estimator_
        w = optmized_model.coef_  # 当为多分类时，w是2维向量
        weight = np.zeros([w.shape[0], n_features])
        weight[:, mask] = w
        return model, weight

    def training(selftest, train_X, train_y):
        # Classfier is SVC
        svc = lr(class_weight='balanced')
        # svc = svm.SVC(kernel='linear', C=1, class_weight='balanced', random_state=0)
        # svc = svm.SVC(kernel='rbf', class_weight='balanced', random_state=0)
        svc.fit(train_X, train_y)
        return svc

    def testing(selftest, model, test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict, decision

    def save_results(selftest, accuracy, sensitivity, specificity, AUC, 
        decision, prediction, label_validation, preffix):
        # Save performances and others
        performances_to_save = np.array([accuracy, sensitivity, specificity, AUC]).reshape(1,4)
        de_pred_label_to_save = np.vstack([decision.T, prediction.T, label_validation.T]).T
        performances_to_save = pd.DataFrame(performances_to_save, columns=[['Accuracy','Sensitivity', 'Specificity', 'AUC']])
        de_pred_label_to_save = pd.DataFrame(de_pred_label_to_save, columns=[['Decision','Prediction', 'Sorted_Real_Label']])
        
        performances_to_save.to_csv(os.path.join(path_out, preffix + '_Performances.txt'), index=False, header=True)
        de_pred_label_to_save.to_csv(os.path.join(path_out, preffix + '_Decision_prediction_label.txt'), index=False, header=True)
        
    def save_fig(selftest, label_validation, prediction, decision, accuracy, sensitivity, specificity, AUC, outname):
        # Save ROC and Classification 2D figure
        acc, sens, spec, auc = eval_performance(label_validation, prediction, decision, 
                                                accuracy, sensitivity, specificity, AUC,
                                                verbose=0, is_showfig=1, is_savefig=1, 
                                                out_name=os.path.join(path_out, outname),
                                                legend1='Healthy', legend2='Unhealthy')

#
if __name__ == '__main__':
    # =============================================================================
    # All inputs
    data_file = r'D:\workstation_b\Fundation\给黎超.xlsx'
    path_out = r'D:\workstation_b\Fundation'
    # =============================================================================
    
    selftest = ClassifyFourKindOfPersonTrain(data_train_file=r'D:\workstation_b\Fundation\feature_train.npy',
                                             data_validation_file=r'D:\workstation_b\Fundation\feature_validation.npy',
                                             label_train_file=r'D:\workstation_b\Fundation\label_train.npy',
                                             label_validation_file=r'D:\workstation_b\Fundation\label_validation.npy',
                                             path_out=path_out,
                                             is_feature_selection=1)


    selftest.main_function()
