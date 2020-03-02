# -*- coding: utf-8 -*-
"""
Created on Wed sel.decision  5 21:12:49 2018
@author: LI Chao
"""

import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN

from Utils.lc_scaler import scaler_apply
from Utils.lc_dimreduction import pca_apply
from Utils.fetch_kfoldidx import fetch_kFold_Index_for_allLabel
from Utils.fetch_kfoldidx import fetch_kfold_idx_for_alllabel_LOOCV
from Utils.lc_featureSelection_rfe import rfeCV
from Utils.lc_evaluation import eval_performance


class SVCRfeCv(object):
    """
    利用递归特征消除的方法筛选特征，然后用SVR训练模型，后用cross-validation的方式来验证
    1、 对特征进行归一化--主成分降维（可选)--RFE--喂入SVC中进行训练--prediction
    2、 采取outer_k-fold的策略
    3、 请注意： 在交叉验证时，我将每个不同的类别的样本都进行split，
       得到训练集和测试集，然后再把每个类别的训练集组合成一个
       大的训练集，测试集同样如此。因此，K参数不能大于数量较少的那个类别
       （比如病人这一类别的样本量是50，正常类是40，那么K不能大于40，且当K=40时，将执行LOOCV）

    Parameters:
    ----------
                 outer_k=3:outer_k-fold
                 step=0.1: rfe step 10%
                 num_jobs=1: parallel
                 scale_method='StandardScaler':standardization method
                 pca_n_component=0.9
                 permutation=0
    Returns:
                各种分类效果等
    """

    def __init__(sel,
                 outer_k=5,
                 scale_method='StandardScaler',
                 pca_n_component=1,  # not use PCA  by default
                 inner_k=5,  # nest k
                 step=0.1,
                 show_results=1,
                 show_roc=0,
                 num_jobs=1,
                 _seed=666,
                 is_resample=True):

        sel.outer_k = outer_k
        sel.scale_method = scale_method
        sel.pca_n_component = pca_n_component
        sel.inner_k = inner_k
        sel.step = step
        sel.show_results = show_results
        sel.show_roc = show_roc
        sel.num_jobs = num_jobs
        sel._seed = 666  # keep the make the results comparable
        sel.is_resample = is_resample  # if resample (over- or under-resampling)

        print("SVCRfeCv initiated")

    def svc_rfe_cv(sel, x, y):
        """main function
        """
        print('training model and _predict using ' +
              str(sel.outer_k) + '-fold CV...\n')

        # preprocess the x and y (# transform data into ndarry and reshape the
        # y into 1 d)
        x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.int16)
        y = np.reshape(y, [-1, ])

        # If k-fold or loocv
        # If is k-fold, then the k must be less than the number of the group
        # that has smallest sample
        num_of_label = [np.sum(y == uni_y) for uni_y in np.unique(y)]
        num_of_smallest_label = np.min(num_of_label)
        if sel.outer_k < num_of_smallest_label:
            index_train, index_test = fetch_kFold_Index_for_allLabel(
                x, y, sel.outer_k, sel._seed)
        elif sel.outer_k == len(y):
            index_train, index_test = fetch_kfold_idx_for_alllabel_LOOCV(y)
        else:
            print(
                "outer_k is greater than sample size!\nthe outer_k = {},\
                and the sample size = {}".format(
                    sel.outer_k, num_of_smallest_label))
            return

        sel.predictlabel = pd.DataFrame([])
        sel.decision = pd.DataFrame([])
        sel.y_real_sorted = pd.DataFrame([])
        sel.weight_all = np.zeros([sel.outer_k, int(
            (len(np.unique(y)) * (len(np.unique(y)) - 1)) / 2), x.shape[1]])

        for i in range(sel.outer_k):
            # split
            x_train, y_train = x[index_train[i]], y[index_train[i]]
            
            # up-resample(Only training dataset)
            if sel.is_resample:
                x_train, y_train = sel.resample(x_train, y_train, method='over-sampling-SMOTE')
        
            x_test, y_test = x[index_test[i]], y[index_test[i]]
            np_size = np.size(x_test)
            if np.shape(x_test)[0] == np_size:
                x_test = x_test.reshape(1, np_size)

            # 根据是否为LOOCV来进行不同的concat
            if sel.outer_k < len(y):
                sel.y_real_sorted = pd.concat(
                    [sel.y_real_sorted, pd.DataFrame(y_test)])
            elif sel.outer_k == len(y):
                sel.y_real_sorted = pd.concat(
                    [sel.y_real_sorted, pd.DataFrame([y_test])])
            else:
                print(
                    "outer_k(outer_k fold) is greater than sample size!\
                     the outer_k = {}, and the sample size = {}".format(
                        sel.outer_k, len(y)))
                return

            # scale
            if sel.scale_method:
                x_train, x_test = scaler_apply(x_train, x_test, sel.scale_method)


            # pca
            if 0 < sel.pca_n_component < 1:
                x_train, x_test, trained_pca = pca_apply(
                    x_train, x_test, sel.pca_n_component)
                print(x_train.shape[1])
            else:
                print(x_train.shape[1])
                pass

            # training
            model, weight = sel._training(x_train, y_train,
                                           step=sel.step, cv=sel.inner_k,
                                           n_jobs=sel.num_jobs)

            # fetch orignal weight
            if 0 < sel.pca_n_component < 1:
                weight = trained_pca.inverse_transform(weight)
            sel.weight_all[i, :, :] = weight

            # test
            prd, de = sel._predict(model, x_test)
            prd = pd.DataFrame(prd)
            de = pd.DataFrame(de)
            sel.predictlabel = pd.concat([sel.predictlabel, prd])
            sel.decision = pd.concat([sel.decision, de])

            print('{}/{}\n'.format(i + 1, sel.outer_k))

        # evaluate trained model
        if sel.show_results:
            sel.accuracy, sel.sensitivity, sel.specificity, sel.auc = \
                eval_performance(
                    sel.y_real_sorted.values,
                    sel.predictlabel.values,
                    sel.decision.values,
                    sel.show_roc)
        return sel

    def resample(sel, data, label, method='over-sampling-SMOTE'):
        """
        Resamle data: over-sampling OR under-sampling
        TODO: Other resample methods.
        """
        if method == 'over-sampling-SMOTE':
            data, label = SMOTE().fit_resample(data, label)
        elif method == 'over-sampling-ADASYN':
            data, label = ADASYN().fit_resample(data, label)
        else:
            print(f'TODO: Other resample methods')
            
        return data, label
        
    def _training(sel, x, y, step, cv, n_jobs):
        model, weight = rfeCV(x, y, step, cv, n_jobs)
        return model, weight

    def _predict(sel, model, test_X):
        predictlabel = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predictlabel, decision


# for debugging
if __name__ == '__main__':
    from sklearn import datasets
    import Machine_learning.classfication.lc_svc_rfe_cv_V3 as lsvc
    x, y = datasets.make_classification(n_samples=500, n_classes=3,
                                        n_informative=50, n_redundant=3,
                                        n_features=100, random_state=1)
    sel = lsvc.SVCRfeCv(outer_k=5)
    
    results = sel.svc_rfe_cv(x, y)
    if results:
        results = results.__dict__
