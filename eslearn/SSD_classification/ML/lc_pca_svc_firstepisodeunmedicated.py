# -*- coding: utf-8 -*-
"""
Created on 2019/11/20
This script is used to training a  linear svc model using training data,  
and test this model using test data with k-fold cross validation-validation stratage.
@author: LI Chao
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
        data : path str
            path of dataset 1
        NOTE: The first column of the dataset is subject unique index, the second is the diagnosis label(0/1),
            the rest of columns are features.

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
                 data=None,
                 is_dim_reduction=True,
                 components=0.95,
                 cv=5):
        
        sel.data =data 
        sel.is_dim_reduction=is_dim_reduction
        sel.components = components
        sel.cv=cv


    def main_function(sel):
        """
        The training data, validation data and  test data are randomly splited
        """
        print('training model and testing...\n')

        # load data
        data  = np.load(sel.data )


        # Extracting features and label
        features_our_center_550 = data [:,2:]
        label_our_center_550 = data [:,1]
        
        # Generate training data and test data	
        data_all = features_our_center_550
        label_all = label_our_center_550

        # Unique ID

        # KFold Cross Validation
        sel.label_test_all = np.array([], dtype=np.int16)
        train_index = np.array([], dtype=np.int16)
        test_index = np.array([], dtype=np.int16)
        sel.decision = np.array([], dtype=np.int16)
        sel.prediction = np.array([], dtype=np.int16)
        sel.accuracy  = np.array([], dtype=np.float16)
        sel.sensitivity  = np.array([], dtype=np.float16)
        sel.specificity  = np.array([], dtype=np.float16)
        sel.AUC = np.array([], dtype=np.float16)
        sel.coef = []     
        kf = KFold(n_splits=sel.cv, shuffle=True, random_state=0)
        for i, (tr_ind , te_ind) in enumerate(kf.split(data_all)):
            print(f'------{i+1}/{sel.cv}...------\n')
            train_index = np.int16(np.append(train_index, tr_ind))
            test_index = np.int16(np.append(test_index, te_ind))
            feature_train = data_all[tr_ind,:]
            label_train = label_all[tr_ind]
            feature_test = data_all[te_ind,:]
            label_test = label_all[te_ind]
            sel.label_test_all = np.int16(np.append(sel.label_test_all, label_test))

            # resampling training data
            # feature_train, label_train = sel.re_sampling(feature_train, label_train)

            # normalization
            prep = elprep.Preprocessing(data_preprocess_method='StandardScaler', data_preprocess_level='subject')
            feature_train, feature_test = prep.data_preprocess(feature_train, feature_test)

            # dimension reduction
            if sel.is_dim_reduction:
                feature_train,feature_test, model_dim_reduction= sel.dimReduction(feature_train, feature_test, sel.components)
                print(f'After dimension reduction, the feature number is {feature_train.shape[1]}')
            else:
                print('No dimension reduction perfromed\n')
            
            # train and test
            print('training and testing...\n')
            model = sel.training(feature_train,label_train) 
            weight = model.coef_

            if sel.is_dim_reduction:
                sel.coef.append(model_dim_reduction.inverse_transform(weight))  # save coef
            else:
                sel.coef.append(weight)  # save coef
                
            pred, dec = sel.testing(model,feature_test)
            sel.prediction = np.append(sel.prediction, np.array(pred))
            sel.decision = np.append(sel.decision, np.array(dec))

            # Evaluating classification performances
            acc, sens, spec, auc = eval_performance(label_test, pred, dec, 
                accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                 verbose=1, is_showfig=0)
        
            sel.accuracy  = np.append(sel.accuracy,acc)
            sel.sensitivity  = np.append(sel.sensitivity,sens)
            sel.specificity  = np.append(sel.specificity,spec)
            sel.AUC = np.append(sel.AUC,auc)
        sel.special_result = np.concatenate([sel.label_test_all, sel.decision, sel.prediction], axis=0).reshape(3, -1).T
        print('Done!')
        return  sel
	
    def re_sampling(sel,feature, label):
        """
        Used to over-sampling unbalanced data
        """
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        feature_resampled, label_resampled = ros.fit_resample(feature, label)
        from collections import Counter
        print(sorted(Counter(label).items()))
        print(sorted(Counter(label_resampled).items()))
        return feature_resampled, label_resampled

    def dimReduction(sel,train_X,test_X, pca_n_component):
        train_X,trained_pca = dimreduction.pca(train_X, pca_n_component)
        test_X=trained_pca.transform(test_X)
        return train_X,test_X, trained_pca
    
    def training(sel,train_X,train_y):
        # svm GrigCV
        svc = svm.SVC(kernel='linear', C=1, class_weight='balanced', max_iter=5000, random_state=0)
        svc.fit(train_X, train_y)
        return svc
    
    def testing(sel,model,test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict,decision

    def save_results(sel, data, name):
        import pickle
        with open(name, 'wb') as f:
            pickle.dump(data, f)
            
    def save_fig(sel, out_name):
        # Save ROC and Classification 2D figure
        acc, sens, spec, auc = eval_performance(sel.label_test_all, sel.prediction, sel.decision, 
                                                sel.accuracy, sel.sensitivity, sel.specificity, sel.AUC,
                                                verbose=0, is_showfig=1, legend1='HC', legend2='SSD', is_savefig=1, 
                                                out_name=out_name)
#        
if __name__=='__main__':
    sel=PCASVCPooling(data=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_firstepisode_and_unmedicated_550.npy')
    
    sel=sel.main_function()
    # sel.save_fig(out_name=r'D:\WorkStation_2018\SZ_classification\Figure\Classification_performances_unmedicated.pdf')

    results=sel.__dict__
    sel.save_results(results, r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_unmedicated_and_firstepisode_550.npy')

    print(np.mean(sel.accuracy))
    print(np.std(sel.accuracy))

    print(np.mean(sel.sensitivity))
    print(np.std(sel.sensitivity))

    print(np.mean(sel.specificity))
    print(np.std(sel.specificity))
    
    print(np.mean(sel.AUC))
    print(np.std(sel.AUC))
    

