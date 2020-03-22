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

from el_classify_sensitive_person_train_validation import ClassifyFourKindOfPersonTrain
from eslearn.utils.lc_evaluation_model_performances import eval_performance


class ClassifyFourKindOfPersonTest():
    """
    This class is used to testing classification model for 2 kind of sensitive person identification.

    Parameters
    ----------
    data_test_file: path str 
        Path of the dataset

    label_test_file: path str 
        Path of the label

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
                 data_test_file=None,
                 label_test_file=None,
                 data_train_file=None,
                 models_path=None,
                 path_out=None,
                 is_feature_selection=False,
                 is_showfig_finally=True):

         selftest.data_test_file = data_test_file
         selftest.label_test_file = label_test_file
         selftest.data_train_file = data_train_file
         selftest.path_out = path_out
         selftest.models_path = models_path
         selftest.is_feature_selection = is_feature_selection
         selftest.is_showfig_finally = is_showfig_finally


    def main_function(selftest):
        """
        """
        print('Training model and testing...\n')

        # load data and mask
        mask_lassocv =  joblib.load(os.path.join(selftest.path_out, 'mask_selected_features_lassocv.pkl'))
        model_feature_selection = joblib.load(os.path.join(selftest.models_path, 'model_feature_selection.pkl'))
        model_classification = joblib.load(os.path.join(selftest.models_path, 'model_classification.pkl'))
        feature_test, selftest.label_test, feature_train = selftest._load_data()  

        # Age encoding
        feature_test[:,2] = ClassifyFourKindOfPersonTrain().age_encodeing(feature_train[:,2], feature_test[:,2])

        # Feature selection
        if selftest.is_feature_selection:   
            feature_test = feature_test[:, mask_lassocv != 0]
            
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
        data_train = np.load(selftest.data_train_file)
        return data_test,  label_test, data_train

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
        
        performances_to_save.to_csv(os.path.join(selftest.path_out, 'test_Performances.txt'), index=False, header=True)
        de_pred_label_to_save.to_csv(os.path.join(selftest.path_out, 'test_Decision_prediction_label.txt'), index=False, header=True)
        
    def save_fig(selftest):
        # Save ROC and Classification 2D figure
        acc, sens, spec, auc = eval_performance(selftest.label_test, selftest.prediction, selftest.decision, 
                                                selftest.accuracy, selftest.sensitivity, selftest.specificity, selftest.AUC,
                                                verbose=0, is_showfig=selftest.is_showfig_finally, is_savefig=1, 
                                                out_name=os.path.join(selftest.path_out, 'Classification_performances_test.pdf'),
                                                legend1='Healthy', legend2='Unhealthy')

#
if __name__ == '__main__':
    # =============================================================================
    # All inputs
    data_file = r'D:\workstation_b\Fundation\给黎超.xlsx'
    path_out = r'D:\workstation_b\Fundation'
    models_path = r'D:\workstation_b\Fundation'
    # =============================================================================
    
    selftest = ClassifyFourKindOfPersonTest(data_test_file=r'D:\workstation_b\Fundation\feature_test.npy',
                                            label_test_file=r'D:\workstation_b\Fundation\label_test.npy',
                                            data_train_file=r'D:\workstation_b\Fundation\feature_train.npy',
                                            path_out=path_out,
                                            models_path=models_path,
                                            is_feature_selection=1)


    selftest.main_function()
