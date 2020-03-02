# -*- coding: utf-8 -*-
import lc_svc_rfe_cv_V2 as lsvc
import pandas as pd
import numpy as np
from lc_read_nii import read_sigleNii_LC
from lc_read_nii import main
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
"""
Created on Wed Dec  5 20:04:02 2018
用神经影像数据作为特征来分类
1 单中心交叉验证
2 多中心之间交叉验证
@author: lenovo
"""
# import
import sys
sys.path.append(
    r'D:\My_Codes\LC_Machine_Learning\Machine_learning (Python)\Machine_learning\utils')
sys.path.append(
    r'D:\My_Codes\LC_Machine_Learning\Machine_learning (Python)\Machine_learning\classfication')
sys.path.append(
    r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\utils')
sys.path.append(
    r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\classfication')


# ==============================================================================
# input

# 外部数据
folder_p_2 = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\C_Weighted_selected'
folder_hc_2 = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_patient\P_Weighted_selected'

# 内部数据
folder_p_1 = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\C_Weighted_selected'
folder_hc_1 = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_patient\P_Weighted_selected'

# 灰质mask
mask = r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'
mask = read_sigleNii_LC(mask) >= 0.2
mask = np.array(mask).reshape(-1,)

# 设置训练与否
if_training_inner_cv = 1
if_training_outer_cv = 0
if_show_data_distribution = 0  # 显示训练集和测试集数据分布
# ==============================================================================


def load_nii_and_gen_label(folder_p, folder_hc, mask):

    # data
    data_p = main(folder_p)
    data_p = np.squeeze(
        np.array([np.array(data_p).reshape(1, -1) for data_p in data_p]))

    data_hc = main(folder_hc)
    data_hc = np.squeeze(
        np.array([np.array(data_hc).reshape(1, -1) for data_hc in data_hc]))

    data = np.vstack([data_p, data_hc])

    # data in mask
#    mask=np.sum(data==0,0)<=0
    data_in_mask = data[:, mask]

    # label
    label = np.hstack(
        [np.ones([len(data_p), ]), np.ones([len(data_hc), ]) - 2])

    return data, data_in_mask, label


data_1, zdata_in_mask_1, label_1 = load_nii_and_gen_label(
    folder_p_1, folder_hc_1, mask)
data_2, zdata_in_mask_2, label_2 = load_nii_and_gen_label(
    folder_p_2, folder_hc_2, mask)

# ===============================================================================
# 检查训练集和测试集的数据一致性
# mean_data_in_mask_1=np.mean(data_in_mask_1,axis=0)
# mean_data_in_mask_2=np.mean(data_in_mask_2,axis=0)
# 结果发现整体来说 数据集1>数据集2

# 尝试被试水平的z标准化
#zdata_in_mask_1=[(data_in_mask_1[i,:]-data_in_mask_1[i,:].mean())/data_in_mask_1[i,:].std() for i in range(data_in_mask_1.shape[0])]
#zdata_in_mask_2=[(data_in_mask_2[i,:]-data_in_mask_2[i,:].mean())/data_in_mask_2[i,:].std() for i in range(data_in_mask_2.shape[0])]

# 尝试被试水平去中心化
#zdata_in_mask_1=[(data_in_mask_1[i,:]-data_in_mask_1[i,:].mean()) for i in range(data_in_mask_1.shape[0])]
#zdata_in_mask_2=[(data_in_mask_2[i,:]-data_in_mask_2[i,:].mean()) for i in range(data_in_mask_2.shape[0])]

# 尝试被试水平除以均值
#zdata_in_mask_1=[(data_in_mask_1[i,:]/data_in_mask_1[i,:].mean()) for i in range(data_in_mask_1.shape[0])]
#zdata_in_mask_2=[(data_in_mask_2[i,:]/data_in_mask_2[i,:].mean()) for i in range(data_in_mask_2.shape[0])]


if if_show_data_distribution:
    zdata_in_mask_1 = pd.DataFrame(data_in_mask_1).values
    zdata_in_mask_2 = pd.DataFrame(data_in_mask_2).values

    mean_1 = np.mean(zdata_in_mask_1, axis=0)
    mean_2 = np.mean(zdata_in_mask_2, axis=0)

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(
        x=mean_1, bins='auto', rwidth=0.7, alpha=0.5, color='b')
    n, bins, patches = ax.hist(
        x=mean_2, bins='auto', rwidth=0.7, alpha=0.5, color='r')
    ax.legend({'our data distribution', 'beijing data distributio'})

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.show()

    plt.savefig(r'J:\dynamicALFF\Results\static_ALFF\test\数据分布_zscore.tif')
# ==============================================================================

# training and test


svc = lsvc.svc_rfe_cv(pca_n_component=0.95, show_results=1, show_roc=0, k=5)

# 单中心内部交叉验证
if if_training_inner_cv:
    results2 = svc.main_svc_rfe_cv(zdata_in_mask_2, label_2)
    results2 = results2.__dict__

    results1 = svc.main_svc_rfe_cv(zdata_in_mask_1, label_1)
    results1 = results1.__dict__

# 多中心之间交叉验证


def cv_multicent(data_in_mask_1, data_in_mask_2, label_1, label_2):
    # 训练2，测试1

    # scale
    data_in_mask_2_standarded, data_in_mask_1_standarded = svc.scaler(
        data_in_mask_2, data_in_mask_1, svc.scale_method)

#    mean_data_in_mask_1_standarded=np.mean(data_in_mask_1_standarded,axis=0)
#    mean_data_in_mask_2_standarded=np.mean(data_in_mask_2_standarded,axis=0)
#    a=mean_data_in_mask_1_standarded-mean_data_in_mask_2_standarded
    # pca
    data_in_mask_2_low_dim, data_in_mask_1_low_dim, trained_pca = svc.dimReduction(
        data_in_mask_2_standarded, data_in_mask_1_standarded, svc.pca_n_component)

    # train
    model, weight = svc.training(data_in_mask_2_low_dim, label_2,
                                 step=svc.step, cv=svc.k, n_jobs=svc.num_jobs,
                                 permutation=svc.permutation)

    # test
    prd, de = svc.testing(model, data_in_mask_1_low_dim)

    # performances

    accuracy = accuracy_score(label_1, prd)
    report = classification_report(label_1, prd)
    report = report.split('\n')
    specificity = report[2].strip().split(' ')
    sensitivity = report[3].strip().split(' ')
    specificity = float([spe for spe in specificity if spe != ''][2])
    sensitivity = float([sen for sen in sensitivity if sen != ''][2])

    # roc and self.auc
    fpr, tpr, thresh = roc_curve(label_1, de)
    auc = roc_auc_score(label_1, de)

    print('\naccuracy={:.2f}\n'.format(accuracy))
    print('sensitivity={:.2f}\n'.format(sensitivity))
    print('specificity={:.2f}\n'.format(specificity))
    print('auc={:.2f}\n'.format(auc))

    return prd, de


if if_training_outer_cv:
    # 训练外部，测试内部
    prd_2to1, de_2to1 = cv_multicent(
        zdata_in_mask_1, zdata_in_mask_2, label_1, label_2)

    # 训练外部，测试内部
    prd_1to2, de_1to2 = cv_multicent(
        zdata_in_mask_2, zdata_in_mask_1, label_2, label_1)
