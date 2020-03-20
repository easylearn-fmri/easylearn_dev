# -*- coding: utf-8 -*-
"""
Created on 2019/11/20
This script is used to training a linear svc model using training dataset, 
and test this model using test dataset with leave-one site cross-validation stratage.

Classfier: Linear SVC
Dimension reduction: PCA

@author: LI Chao
Email: lichao19870617@gmail.com
"""

import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
# sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import Utils.lc_dimreduction as dimreduction
from Utils.lc_evaluation_model_performances import eval_performance


class SVCRFECV():
    def __init__(sel,
                 data_550_path=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_550.npy',
                 data_train_path=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_206.npy',
                 data_COBRE_path=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_COBRE.npy',
                 data_UCAL_path=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_UCLA.npy',
                 is_dim_reduction=1,
                 components=0.95,
                 cv=5,
                 show_results=1,
                 show_roc=1):

        sel.data_550_path = data_550_path
        sel.data_train_path = data_train_path
        sel.data_COBRE_path = data_COBRE_path
        sel.data_UCAL_path = data_UCAL_path

        sel.is_dim_reduction = is_dim_reduction
        sel.components = components
        sel.cv = cv
        sel.show_results = show_results
        sel.show_roc = show_roc

    def main_svc_rfe_cv(sel):
        print('Training model and testing...\n')
        # Load data
        feature_550, label_550 = sel._load_data(sel.data_550_path)
        feature_206, label_206 = sel._load_data(sel.data_train_path)
        feature_COBRE, label_COBRE = sel._load_data(sel.data_COBRE_path)
        feature_UCAL, label_UCAL = sel._load_data(sel.data_UCAL_path)
        feature_all = [feature_550, feature_206, feature_COBRE, feature_UCAL]
        label_all = [label_550, label_206, label_COBRE, label_UCAL]

        # Leave one site CV
        n_site = len(label_all)
        name = ['550','206','COBRE','UCLA']
        sel.decision = np.array([], dtype=np.int16)
        sel.prediction = np.array([], dtype=np.int16)
        sel.accuracy = np.array([], dtype=np.float16)
        sel.sensitivity = np.array([], dtype=np.float16)
        sel.specificity = np.array([], dtype=np.float16)
        sel.AUC = np.array([], dtype=np.float16)
        sel.coef = []
        for i in range(n_site):
            print('-'*40)
            print(f'{i+1}/{n_site}: test dataset is {name[i]}...')
            feature_train, label_train = feature_all.copy(), label_all.copy()
            feature_test, label_test = feature_train.pop(i), label_train.pop(i)
            feature_train = np.concatenate(feature_train, axis=0)
            label_train = np.concatenate(label_train, axis=0)

            # Resampling training data
            feature_train, label_train = sel.re_sampling(
                feature_train, label_train)

            # Normalization
            feature_train = sel.normalization(feature_train)
            feature_test = sel.normalization(feature_test)

            # Dimension reduction
            if sel.is_dim_reduction:
                feature_train, feature_test, model_dim_reduction = sel.dimReduction(
                    feature_train, feature_test, sel.components)
                print(f'After dimension reduction, the feature number is {feature_train.shape[1]}')
            else:
                print('No dimension reduction perfromed\n')

            # Train and test
            print('training and testing...\n')
            model = sel.training(feature_train, label_train, sel.cv)
            if sel.is_dim_reduction:
                sel.coef.append(model_dim_reduction.inverse_transform(model.coef_))  # save coef
            else:
                sel.coef.append(model.coef_)  # save coef

            pred, dec = sel.testing(model, feature_test)
            sel.prediction = np.append(sel.prediction, np.array(pred))
            sel.decision = np.append(sel.decision, np.array(dec))
            
            # Evaluating classification performances
            acc, sens, spec, auc = eval_performance(label_test, pred, dec, 0)
            sel.accuracy = np.append(sel.accuracy, acc)
            sel.sensitivity = np.append(sel.sensitivity, sens)
            sel.specificity = np.append(sel.specificity, spec)
            sel.AUC = np.append(sel.AUC, auc)
            print(f'performances = {acc, sens, spec,auc}')
        return sel

    def _load_data(sel, data_path):
        data = np.load(data_path)
        name = data_path.split('\\')[-1]
        print(f"Dataset is {name}")
        feature = data[:, 2:]
        label = data[:, 1]
        return feature, label

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

    def dimReduction(sel, train_X, test_X, pca_n_component):
        train_X, trained_pca = dimreduction.pca(train_X, pca_n_component)
        test_X = trained_pca.transform(test_X)
        return train_X, test_X, trained_pca

    def training(sel, train_X, train_y, cv):
        # svm GrigCV
        svc = svm.LinearSVC(class_weight='balanced')
        # svc = svm.SVC(class_weight='balanced')
        # svc = GridSearchCV(svc, param, cv=cv)
        svc.fit(train_X, train_y)
        return svc

    def testing(sel, model, test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict, decision

    def save_results(sel, data, name):
        import pickle
        with open(name, 'wb') as f:
            pickle.dump(data, f)


#
if __name__ == '__main__':
    sel = SVCRFECV()
    results = sel.main_svc_rfe_cv()
    
    results = results.__dict__
    sel.save_results(results, r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\results_leave_one_site_cv')

    print(np.mean(results['accuracy']))
    print(np.std(results['accuracy']))

    print(np.mean(results['sensitivity']))
    print(np.std(results['sensitivity']))

    print(np.mean(results['specificity']))
    print(np.std(results['specificity']))

    print(np.mean(results['AUC']))
    print(np.std(results['AUC']))
