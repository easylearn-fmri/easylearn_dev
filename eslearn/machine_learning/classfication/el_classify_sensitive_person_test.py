# -*- coding: utf-8 -*-
"""
Created on 2020/03/16
Feature selection: Relief-based feature selection algorithm.
------
@author: LI Chao
"""

import numpy as np
from sklearn import preprocessing
import os
from sklearn.externals import joblib
from eslearn.utils.lc_evaluation_model_performances import eval_performance


class ClassifyFourKindOfPersonTest():
    """
    This class is used to training classification model for 4 kind of person identification..
    Muticlass classification.

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

    Returns
    -------
    Save all classification results and figures to local disk.
    """
    def __init__(selftest,
                 data_test_file=None,
                 label_test_file=None,
                 models_path=None,
                 path_out=None,
                 is_feature_selection=False,
                 is_showfig_finally=True,
                 rand_seed=666):

         selftest.data_test_file = data_test_file
         selftest.label_test_file = label_test_file
         selftest.path_out = path_out
         selftest.models_path = models_path
         selftest.is_feature_selection = is_feature_selection
         selftest.is_showfig_finally = is_showfig_finally
         selftest.rand_seed = rand_seed


    def main_function(selftest):
        """
        """
        print('Training model and testing...\n')

        # load data and mask
        model_feature_selection = joblib.load(os.path.join(selftest.models_path, 'model_feature_selection.pkl'))
        model_classification = joblib.load(os.path.join(selftest.models_path, 'model_classification.pkl'))
        feature_test, selftest.label_test = selftest._load_data()
        
        # data_preprocess_in_group_level
        feature_test = selftest.data_preprocess_in_subject_level(feature_test)   

        # Feature selection
        if selftest.is_feature_selection:   
            feature_test = model_feature_selection.transform(feature_test)     

        # Testting
        selftest.prediction, selftest.decision = selftest.testing(model_classification, feature_test)

        # Evaluating classification performances
        selftest.accuracy, selftest.sensitivity, selftest.specificity, selftest.AUC = eval_performance(selftest.label_test, selftest.prediction, selftest.decision, 
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
             verbose=1, is_showfig=0)

        # Save results and fig to local path
        selftest.save_results()
        selftest.save_fig()
            
        print("--" * 10 + "Done!" + "--" * 10 )
        return selftest


    def _load_data(selftest):
        """
        Load data
        """
        data_test = np.load(selftest.data_test_file)
        label_test = np.load(selftest.label_test_file)
        return data_test,  label_test


    def data_preprocess_in_subject_level(selftest, feature):
        '''
        This function is used to preprocess features in subject level.
        '''
        scaler = preprocessing.StandardScaler().fit(feature.T)
        feature = scaler.transform(feature.T) .T
        return feature

    def testing(selftest, model, test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict, decision

    def save_results(selftest):
        # Save performances and others
        import pandas as pd
        performances_to_save = np.array([selftest.accuracy, selftest.sensitivity, selftest.specificity, selftest.AUC]).reshape(1,4)
        de_pred_label_to_save = np.vstack([selftest.decision.T, selftest.prediction.T, selftest.label_test.T]).T
        performances_to_save = pd.DataFrame(performances_to_save, columns=[['Accuracy','Sensitivity', 'Specificity', 'AUC']])
        de_pred_label_to_save = pd.DataFrame(de_pred_label_to_save, columns=[['Decision','Prediction', 'Sorted_Real_Label']])
        
        performances_to_save.to_csv(os.path.join(selftest.path_out, 'Performances.txt'), index=False, header=True)
        de_pred_label_to_save.to_csv(os.path.join(selftest.path_out, 'Decision_prediction_label.txt'), index=False, header=True)
        
    def save_fig(selftest):
        # Save ROC and Classification 2D figure
        acc, sens, spec, auc = eval_performance(selftest.label_test, selftest.prediction, selftest.decision, 
                                                selftest.accuracy, selftest.sensitivity, selftest.specificity, selftest.AUC,
                                                verbose=0, is_showfig=selftest.is_showfig_finally, is_savefig=1, 
                                                out_name=os.path.join(selftest.path_out, 'Classification_performances.pdf'))

#
if __name__ == '__main__':
    # =============================================================================
    # All inputs
    data_file = r'D:\workstation_b\YiFan\给黎超.xlsx'
    path_out = r'D:\workstation_b\YiFan'
    models_path = r'D:\workstation_b\YiFan'
    # =============================================================================
    
    selftest = ClassifyFourKindOfPersonTest(data_test_file=r'D:\workstation_b\YiFan\feature_test.npy',
                                            label_test_file=r'D:\workstation_b\YiFan\label_test.npy',
                                            path_out=path_out,
                                            models_path=models_path,
                                            is_feature_selection=1)


    selftest.main_function()
