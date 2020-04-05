# -*- coding: utf-8 -*-
"""
Created on 2019/11/20
This script is used to training a  linear svc model using training dataset, 
and test this model using test dataset with pooling cross-validation stratage.

All datasets (4 datasets) were concatenate into one single dataset, then using cross-validation strategy.

Classfier: Linear SVC
Dimension reduction: PCA

@author: LI Chao
Email: lichao19870617@gmail.com
"""
import sys
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import preprocessing

import eslearn.utils.el_preprocessing as elprep
import eslearn.utils.lc_dimreduction as dimreduction
from eslearn.utils.lc_evaluation_model_performances import eval_performance

class PCASVCPooling():
    """
    Parameters:
    ----------
        dataset_our_center_550 : path str
            path of dataset 1. 
            NOTE: The first column of the dataset is subject unique index, the second is the diagnosis label(0/1),
            the rest of columns are features. The other dataset are the same as this dataset.

        dataset_206: path str
            path of dataset 2

        dataset_COBRE: path str
            path of dataset 3

        dataset_UCAL: path str
            path of dataset 4

        is_dim_reduction： bool
            if perform dimension reduction (PCA)

        components: float
            How many percentages of the cumulatively explained variance to be retained. This is used to select the top principal components.

        cv: int
            How many folds of the cross-validation.

        out_name: str
            The name of the output results.

    Returns:
    --------
        Classification results, such as accuracy, sensitivity, specificity, AUC and figures that used to report.

    """
    def __init__(sel,
                 dataset_our_center_550=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_550.npy',
                 dataset_206=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_206.npy',
                 dataset_COBRE=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_COBRE.npy',
                 dataset_UCAL=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_UCLA.npy',
                 is_dim_reduction=True,
                 components=0.95,
                 cv=5):

        sel.dataset_our_center_550 = dataset_our_center_550
        sel.dataset_206 = dataset_206
        sel.dataset_COBRE = dataset_COBRE
        sel.dataset_UCAL = dataset_UCAL

        sel.is_dim_reduction = is_dim_reduction
        sel.components = components
        sel.cv = cv

    def main_function(sel):
        """
        The training data, validation data and  test data are randomly splited
        """
        print('Training model and testing...\n')

        # load data
        dataset_our_center_550 = np.load(sel.dataset_our_center_550)
        dataset_206 = np.load(sel.dataset_206)
        dataset_COBRE = np.load(sel.dataset_COBRE)
        dataset_UCAL = np.load(sel.dataset_UCAL)

        # Extracting features and label
        features_our_center_550 = dataset_our_center_550[:, 2:]
        features_206 = dataset_206[:, 2:]
        features_COBRE = dataset_COBRE[:, 2:]
        features_UCAL = dataset_UCAL[:, 2:]

        label_our_center_550 = dataset_our_center_550[:, 1]
        label_206 = dataset_206[:, 1]
        label_COBRE = dataset_COBRE[:, 1]
        label_UCAL = dataset_UCAL[:, 1]

        # Generate training data and test data
        data_all = np.concatenate(
            [features_our_center_550, features_206, features_UCAL, features_COBRE], axis=0)
        label_all = np.concatenate(
            [label_our_center_550, label_206, label_UCAL, label_COBRE], axis=0)

        # Unique ID
        uid_our_center_550 = np.int32(dataset_our_center_550[:, 0])
        uid_206 = np.int32(dataset_206[:, 0])
        uid_all = np.concatenate([uid_our_center_550, uid_206, 
                                  np.zeros(len(label_UCAL, )) -1, 
                                  np.zeros(len(label_COBRE, )) -1], axis=0)
        uid_all = np.int32(uid_all)

        # KFold Cross Validation
        sel.label_test_all = np.array([], dtype=np.int16)
        train_index = np.array([], dtype=np.int16)
        test_index = np.array([], dtype=np.int16)
        sel.decision = np.array([], dtype=np.int16)
        sel.prediction = np.array([], dtype=np.int16)
        sel.accuracy = np.array([], dtype=np.float16)
        sel.sensitivity = np.array([], dtype=np.float16)
        sel.specificity = np.array([], dtype=np.float16)
        sel.AUC = np.array([], dtype=np.float16)
        sel.coef = []        
        kf = KFold(n_splits=sel.cv, shuffle=True, random_state=0)
        for i, (tr_ind, te_ind) in enumerate(kf.split(data_all)):
            print(f'------{i+1}/{sel.cv}...------\n')
            train_index = np.int16(np.append(train_index, tr_ind))
            test_index = np.int16(np.append(test_index, te_ind))
            feature_train = data_all[tr_ind, :]
            label_train = label_all[tr_ind]
            feature_test = data_all[te_ind, :]
            label_test = label_all[te_ind]
            sel.label_test_all = np.int16(np.append(sel.label_test_all, label_test))

            # resampling training data
            # feature_train, label_train = sel.re_sampling(feature_train, label_train)

            # normalization
            prep = elprep.Preprocessing(data_preprocess_method='StandardScaler', data_preprocess_level='subject')
            feature_train, feature_test = prep.data_preprocess(feature_train, feature_test)

            # dimension reduction
            if sel.is_dim_reduction:
                feature_train, feature_test, model_dim_reduction = sel.dimReduction(
                    feature_train, feature_test, sel.components)
                print(f'After dimension reduction, the feature number is {feature_train.shape[1]}')
            else:
                print('No dimension reduction perfromed\n')
                
            # train
            print('training and testing...\n')
            # model, weight = rfeCV(feature_train, label_train, step=0.2, cv=3, n_jobs=-1, permutation=0)
            model = sel.training(feature_train, label_train)
            coef = model.coef_
            # coef = weight
            
            # Weight
            if sel.is_dim_reduction:
                sel.coef.append(model_dim_reduction.inverse_transform(coef))  # save coef
            else:
                sel.coef.append(coef)  # save coef
                
            # test
            pred, dec = sel.testing(model, feature_test)
            sel.prediction = np.append(sel.prediction, np.array(pred))
            sel.decision = np.append(sel.decision, np.array(dec))

            # Evaluating classification performances
            acc, sens, spec, auc = eval_performance(label_test, pred, dec, 
                accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                 verbose=1, is_showfig=0)
            sel.accuracy = np.append(sel.accuracy, acc)
            sel.sensitivity = np.append(sel.sensitivity, sens)
            sel.specificity = np.append(sel.specificity, spec)
            sel.AUC = np.append(sel.AUC, auc)

        uid_all_sorted = np.int32(uid_all[test_index])
        sel.special_result = np.concatenate(
            [uid_all_sorted, sel.label_test_all, sel.decision, sel.prediction], axis=0).reshape(4, -1).T
        print('Done!')
        return sel

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

    def dimReduction(sel, train_X, test_X, pca_n_component):
        train_X, trained_pca = dimreduction.pca(train_X, pca_n_component)
        test_X = trained_pca.transform(test_X)
        return train_X, test_X, trained_pca

    def training(sel, train_X, train_y):  
        svc = svm.SVC(kernel='linear', C=1, class_weight='balanced',
                      max_iter=5000, random_state=0)
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
            
    def save_fig(sel, out_name):
        # Save ROC and Classification 2D figure
        acc, sens, spec, auc = eval_performance(sel.label_test_all, sel.prediction, sel.decision, 
                                                sel.accuracy, sel.sensitivity, sel.specificity, sel.AUC,
                                                verbose=0, is_showfig=1, legend1='HC', legend2='SZ', is_savefig=1, 
                                                out_name=out_name)
#
if __name__ == '__main__':
    sel=PCASVCPooling()
    
    results=sel.main_function()
    sel.save_fig(out_name=r'D:\WorkStation_2018\SZ_classification\Figure\Classification_performances_pooling.pdf')
    results=results.__dict__
    sel.save_results(results, r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_pooling.npy')
    
    print(np.mean(sel.accuracy))
    print(np.std(sel.accuracy))

    print(np.mean(sel.sensitivity))
    print(np.std(sel.sensitivity))

    print(np.mean(sel.specificity))
    print(np.std(sel.specificity))
    
    print(np.mean(sel.AUC))
    print(np.std(sel.AUC))
