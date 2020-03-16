# -*- coding: utf-8 -*-
"""
Created on 2020/03/16
------
@author: LI Chao
"""

import numpy as np
import pandas as pd
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn import  preprocessing
import os
from sklearn.externals import joblib
from sklearn.linear_model import lasso_path, enet_path

from eslearn.feature_selection.el_rfe import rfeCV
from eslearn.utils.lc_evaluation_model_performances import eval_performance
from eslearn.utils.el_preprocessing import Preprocessing
from eslearn.utils.lc_niiProcessor import NiiProcessor
import eslearn.utils.el_preprocessing as elprep


class ClassifyFourKindOfPersonTrain():
    """
    This class is used to training classification model for 2 kind of sensitive person identification.
    The output model can be trained using given initial weights.

    Parameters
    ----------
    data_file: path str 
        Path of the dataset

    path_out : 
        Path to save results

    data_preprocess_method: str
        How to preprocess features 'StandardScaler' OR 'MinMaxScaler'.
        
    data_preprocess_level: str
        Which level to preprocess features. 'group' or 'subject'
        
    is_dim_reduction : bool
        If perfrome dimension reduction.

    is_feature_selection : bool
        if perfrome feature selection.

    n_features_to_select: int
        number of features to select

    n_components: float from 0 to 1
        If is_dim_reduction, then how many components to remain.

    num_of_kfold: int
        Number of the k in k-fold cross-validation

    is_showfig_finally: bool
        If show figure after all iteration finished.

    is_showfig_in_each_fold: bool
        If show figure in each fold.

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
        feature_train, feature_validation, selftest.label_train, selftest.label_validation = selftest._load_data()
        
        # Onehot encoding
        # catg_id = np.arange(0, np.shape(feature_train)[1])
        # catg_id = np.delete(catg_id, 2)
        # onehot = OneHotEncoder()
        # onehot.fit(feature_train[:, catg_id])
        # feature_train_cat = onehot.transform(feature_train[:, catg_id]).toarray()
        # feature_validation_cat= onehot.transform(feature_validation[:, catg_id]).toarray()
        # feature_train = np.hstack([feature_train[:,2].reshape(-1,1), feature_train_cat])
        # feature_validation = np.hstack([feature_validation[:,2].reshape(-1,1), feature_validation_cat])

        # data_preprocess_in_group_level
        # feature_train = selftest.data_preprocess_in_subject_level(feature_train,)
        # feature_validation = selftest.data_preprocess_in_subject_level(feature_validation)    

        # Feature selection
        if selftest.is_feature_selection:
            feature_train, feature_validation, mask, n_features_origin, fs = \
                  selftest.feature_selection_relief(feature_train, 
                  selftest.label_train, feature_validation, selftest.n_features_to_select)   
            with open(os.path.join(selftest.path_out, 'model_feature_selection.pkl'), 'wb') as f_fs:
                joblib.dump(fs, f_fs)
                
        # Train
        print('training and testing...\n')
        if selftest.is_feature_selection:
            model = selftest.training(feature_train, selftest.label_train)
        else:
            model = selftest.training(feature_train, selftest.label_train)
            # model, w = selftest.rfeCV_training(feature_train, selftest.label_train)

        # Save model
        with open(os.path.join(selftest.path_out, 'model_classification.pkl'), 'wb') as f_model:
            joblib.dump(model, f_model)
        
        # Get weight 
        # if selftest.is_feature_selection:
        #     coef = np.zeros([n_features_origin,])
        #     coef[mask] = model.coef_
        # else:
        #     coef = model.coef_
        
        # Validating
        selftest.prediction, selftest.decision = selftest.testing(model, feature_validation)

        # Evaluating classification performances
        selftest.accuracy, selftest.sensitivity, selftest.specificity, selftest.AUC = eval_performance(selftest.label_validation,selftest.prediction, selftest.decision, 
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
             verbose=1, is_showfig=0)

        # Save results and fig to local path
        selftest.save_results()
        selftest.save_fig()
        
        print(f"MSE = {np.mean(np.power((selftest.decision - selftest.label_validation), 2))}")
            
        print("--" * 10 + "Done!" + "--" * 10 )
        return selftest


    def _load_data(selftest):
        """
        Load data
        """
        data_train = np.load(selftest.data_train_file)[:, np.array([0, 8, 13, 15])]
        data_validation = np.load(selftest.data_validation_file)[:, np.array([0, 8, 13, 15])]
        # data_train = np.load(selftest.data_train_file)
        # data_validation = np.load(selftest.data_validation_file)
        label_train = np.load(selftest.label_train_file)
        label_validation = np.load(selftest.label_validation_file)
        return data_train, data_validation, label_train, label_validation
    

    def re_sampling(selftest, feature, label):
        """
        Used to over-sampling unbalanced data
        """
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        feature_resampled, label_resampled = ros.fit_resample(feature, label)
        from collections import Counter
        print(f"After re-sampling, the sample size are: {sorted(Counter(label_resampled).items())}")
        return feature_resampled, label_resampled

    def data_preprocess_in_subject_level(selftest, feature):
        '''
        This function is used to preprocess features in subject level.
        '''
        scaler = preprocessing.StandardScaler().fit(feature.T)
        # scaler = preprocessing.MinMaxScaler().fit(feature.T)
        feature = scaler.transform(feature.T) .T
        return feature
    
    def feature_selection_lasso(selftest, feature, label):
        model = LassoCV(cv=5)
        model.fit(feature, label)
        return model.coef_ != 0
        
        
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

    def rfeCV_training(selftest, train_X, train_y, step=4, num_fold_of_inner_rfeCV=10, n_jobs=-1):
        model, weight = rfeCV(train_X, train_y, step, num_fold_of_inner_rfeCV, n_jobs)
        return model, weight

    def training(selftest, train_X, train_y):
        # Classfier is SVC
        svc = lr(class_weight='balanced')
        # svc = svm.SVC(kernel='linear', C=1, class_weight='balanced', random_state=0)
        svc.fit(train_X, train_y)
        return svc

    def testing(selftest, model, test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict, decision

    def save_results(selftest):
        # Save performances and others
        performances_to_save = np.array([selftest.accuracy, selftest.sensitivity, selftest.specificity, selftest.AUC]).reshape(1,4)
        de_pred_label_to_save = np.vstack([selftest.decision.T, selftest.prediction.T, selftest.label_validation.T]).T
        performances_to_save = pd.DataFrame(performances_to_save, columns=[['Accuracy','Sensitivity', 'Specificity', 'AUC']])
        de_pred_label_to_save = pd.DataFrame(de_pred_label_to_save, columns=[['Decision','Prediction', 'Sorted_Real_Label']])
        
        performances_to_save.to_csv(os.path.join(selftest.path_out, 'Performances.txt'), index=False, header=True)
        de_pred_label_to_save.to_csv(os.path.join(selftest.path_out, 'Decision_prediction_label.txt'), index=False, header=True)
        
    def save_fig(selftest):
        # Save ROC and Classification 2D figure
        acc, sens, spec, auc = eval_performance(selftest.label_validation, selftest.prediction, selftest.decision, 
                                                selftest.accuracy, selftest.sensitivity, selftest.specificity, selftest.AUC,
                                                verbose=0, is_showfig=selftest.is_showfig_finally, is_savefig=1, 
                                                out_name=os.path.join(selftest.path_out, 'Classification_performances.pdf'))

#
if __name__ == '__main__':
    # =============================================================================
    # All inputs
    data_file = r'D:\workstation_b\YiFan\给黎超.xlsx'
    path_out = r'D:\workstation_b\YiFan'
    # =============================================================================
    
    selftest = ClassifyFourKindOfPersonTrain(data_train_file=r'D:\workstation_b\YiFan\feature_train.npy',
                                             data_validation_file=r'D:\workstation_b\YiFan\feature_validation.npy',
                                             label_train_file=r'D:\workstation_b\YiFan\label_train.npy',
                                             label_validation_file=r'D:\workstation_b\YiFan\label_validation.npy',
                                             path_out=path_out,
                                             is_feature_selection=0,
                                             n_features_to_select=4)


    selftest.main_function()
