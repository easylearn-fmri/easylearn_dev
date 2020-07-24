# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:43:03 2018
anova for feature selection and svm for classfication
@author: LI Chao (0.7279411764705882, 0.66, 0.79, 0.8419117647058824)
(0.7132352941176471, 0.66, 0.76, 0.8380190311418685)
"""
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root)
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report


class AnovaSvm():

    def __init__(sel):
        sel.n_selectedFeatures = 6
        sel.kernel = 'linear'
        sel.class_weight = 'balanced'
        sel.random_state = 888

    def main_anova_svm(sel, x, y):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=sel.random_state)

        # ANOVA SVM-C
        # 1) anova filter, take k best ranked features
        anova_filter = SelectKBest(f_regression, k=sel.n_selectedFeatures)
        # 2) built svm
        clf = svm.SVC(kernel=sel.kernel, class_weight=sel.class_weight)
        anova_svm = make_pipeline(anova_filter, clf)
        anova_svm.fit(x_train, y_train)
        # 3) predict
        sel.y_pred = anova_svm.predict(x_test)
        sel.decision = anova_svm.decision_function(x_test)
        sel.weight = clf.coef_
        sel.selected_features = anova_filter.get_support()

        # eval
        ac = y_test - sel.y_pred
        print('Accuracy={:.2f}\n'.format(sum(ac == 0) / len(ac)))

        print(classification_report(y_test, sel.y_pred))
        return sel


if __name__ == '__main__':
    from Utils.lc_read_write_Mat import read_mat, write_mat
    import pandas as pd
    import numpy as np
    sel = AnovaSvm()


# =============================================================================
#     # 生成数据
#     x, y = samples_generator.make_classification(
#     n_samples=200,n_features=20, n_informative=3, n_redundant=0, n_classes=3,
#     n_clusters_per_class=2)
# =============================================================================

    x = read_mat(
        r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\staticFC.mat', dataset_name=None)

    y = pd.read_excel(
        r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\folder_label.xlsx')
    y = y['诊断'].values

    order = [1, 4]

    y1 = np.hstack([y[y == order[0]], y[y == order[1]]])
    x = np.vstack([x[y == order[0], :], x[y == order[1], :]])
    y = y1
    print(sum(y == order[0]), sum(y == order[1]))

#    from sklearn.preprocessing import OneHotEncoder
#    #哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
#    enc=OneHotEncoder()
#    y=enc.fit(y1.reshape(-1,1))

    results = sel.main_anova_svm(x, y)
    results = results.__dict__
