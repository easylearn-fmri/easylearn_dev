# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:15:54 2018
1.以功能连接/动态功能连接矩阵为特征，来进行分类
2.本程序使用的算法为svc（交叉验证）
3.当特征是动态连接时，使用标准差或者均数等来作为特征。也可以自己定义
4.input：
    所有人的.mat FC/dFC
5.output:
    机器学习的相应结果，以字典形式保存再result中。
 
@author: lI Chao
"""
import sys
sys.path.append(
    r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')
sys.path.append(
    r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Machine_learning\classfication')

from sklearn.model_selection import train_test_split
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import re
import os
from lc_read_write_mat import read_mat, write_mat



class classify_using_FC():

    def __init__(sel,
                 k=5,
                 file_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\DynamicFC_length17_step1_screened',
                 dataset_name=None,  # mat文件打开后的名字
                 scale=r'D:\WorkStation_2018\WorkStation_dynamicFC\Scales\8.30大表.xlsx',
                 save_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic',
                 feature='std17',  # 用均数还是std等('mean'/'std'/'staticFC')
                 folder_label_name='folder_label17.xlsx',
                 which_group_to_classify=[2, 3],
                 how_resample='up_resample',  # 样本不匹配时
                 mask=np.ones([114, 114]),  # 特征矩阵的mask
                 n_processess=10,
                 if_save_post_mat=1,  # 保存后处理后的mat？
                 random_state=2):

        sel.k = k
        sel.file_path = file_path
        sel.dataset_name = dataset_name
        sel.scale = scale
        sel.save_path = save_path
        sel.feature = feature
        sel.folder_label_name = folder_label_name
        sel.which_group_to_classify = which_group_to_classify
        sel.how_resample = how_resample

        sel.mask = mask
        sel.mask = np.triu(sel.mask, 1) == 1  # 只提取上三角（因为其他的为重复）
#        sel.mask=np.ones([1,4])==1 # 只提取上三角（因为其他的为重复）

        sel.n_processess = n_processess
        sel.if_save_post_mat = if_save_post_mat

        sel.random_state = random_state

    def load_allmat(sel):
        # 多线程
        s = time.time()
        print('loading all mat...\n')

        # 判断是否有FC mat文件
        if os.path.exists(os.path.join(sel.save_path, sel.feature + '.mat')):
            sel.mat = pd.DataFrame(read_mat(os.path.join(
                sel.save_path, sel.feature + '.mat'), None))
            print('Already have {}\nloaded all mat!\nrunning time={:.2f}'.format(
                sel.feature + '.mat', time.time() - s))
        else:

            sel.all_mat = os.listdir(sel.file_path)
            all_mat_path = [os.path.join(sel.file_path, all_mat_)
                            for all_mat_ in sel.all_mat]

            cores = multiprocessing.cpu_count()
            if sel.n_processess > cores:
                sel.n_processess = cores - 1

            len_all = len(all_mat_path)
            sel.mat = pd.DataFrame([])

            # 特征用std还是mean
            if re.match('mean', sel.feature):
                ith = 1
            elif re.match('std', sel.feature):
                ith = 0
            elif re.match('static', sel.feature):
                ith = 0
            else:
                print('###还未添加其他衡量dFC的指标,默认使用std###\n')
                ith = 0

            # load mat...
            with ThreadPoolExecutor(sel.n_processess) as executor:
                for i, all_mat_ in enumerate(all_mat_path):
                    task = executor.submit(
                        sel.load_onemat_and_processing, i, all_mat_, len_all, s)
                    sel.mat = pd.concat(
                        [sel.mat, pd.DataFrame(task.result()[ith]).T], axis=0)

            # 保存后处理后的mat文件
            if sel.if_save_post_mat:
                write_mat(fileName=os.path.join(sel.save_path, sel.feature + '.mat'),
                          dataset_name=sel.feature,
                          dataset=np.mat(sel.mat.values))
                print('saved all {} mat!\n'.format(sel.feature))

        return sel

    def load_onemat_and_processing(sel, i, all_mat_, len_all, s):
        # load mat
        mat = read_mat(all_mat_, sel.dataset_name)

        # 计算方差，均数等。可扩展。(如果时静态FC，则不执行)
        if re.match('static', sel.feature):
            mat_std, mat_mean = mat, []
        else:
            mat_std, mat_mean = sel.calc_std(mat)

        # 后处理特征，可扩展
        if re.match('static', sel.feature):
            mat_std_1d, mat_mean_1d = sel.postprocessing_features(mat_std), []
        else:
            mat_std_1d = sel.postprocessing_features(mat_std)
            mat_mean_1d = sel.postprocessing_features(mat_mean)

        # 打印load进度
        if i % 20 == 0 or i == 0:
            print('{}/{}\n'.format(i, len_all))

        if i % 50 == 0 and i != 0:
            e = time.time()
            remaining_running_time = (e - s) * (len_all - i) / i
            print('\nremaining time={:.2f} seconds \n'.format(
                remaining_running_time))

        return mat_std_1d, mat_mean_1d

    def calc_std(sel, mat):
        mat_std = np.std(mat, axis=2)
        mat_mean = np.mean(mat, axis=2)
        return mat_std, mat_mean

    def postprocessing_features(sel, mat):
        # 准备特征：比如取上三角，拉直等
        return mat[sel.mask]

    def gen_label(sel):

        # 判断是否已经存在label
        if os.path.exists(os.path.join(sel.save_path, sel.folder_label_name)):
            sel.label = pd.read_excel(os.path.join(
                sel.save_path, sel.folder_label_name))['诊断']
            print('\nAlready have {}\n'.format(sel.folder_label_name))

        else:
            # identify label for each subj
            id_subj = pd.Series(sel.all_mat).str.extract('([1-9]\d*)')

            scale = pd.read_excel(sel.scale)

            id_subj = pd.DataFrame(id_subj, dtype=type(scale['folder'][0]))

            sel.label = pd.merge(
                scale, id_subj, left_on='folder', right_on=0, how='inner')['诊断']
            sel.folder = pd.merge(
                scale, id_subj, left_on='folder', right_on=0, how='inner')['folder']

            # save folder and label
            if sel.if_save_post_mat:
                sel.label_folder = pd.concat([sel.folder, sel.label], axis=1)
                sel.label_folder.to_excel(os.path.join(
                    sel.save_path, sel.folder_label_name), index=False)

        return sel

    def machine_learning(sel):

        # label
        y = pd.concat([sel.label[sel.label.values == sel.which_group_to_classify[0]],
                       sel.label[sel.label.values == sel.which_group_to_classify[1]]])
        y = y.values
        # x/sel.mat
        if os.path.exists(os.path.join(sel.save_path, sel.feature + '.mat')):
            sel.mat = pd.DataFrame(read_mat(os.path.join(
                sel.save_path, sel.feature + '.mat'), None))

        x = pd.concat([sel.mat.iloc[sel.label.values == sel.which_group_to_classify[0], :],
                       sel.mat.iloc[sel.label.values == sel.which_group_to_classify[1], :]])
        x = x.values
        # 二值化y
        y[y == sel.which_group_to_classify[0]] = 0
        y[y == sel.which_group_to_classify[1]] = 1

        # 打印未平衡前的样本
        print('未平衡前的样本={}:{}\n'.format(sum(y == 0), sum(y == 1)))
        sel.origin_sample_size = '{}:{}'.format(sum(y == 0), sum(y == 1))

        # 平衡样本(上采样)
        ind_up, ind_down = np.argmin([sum(y == 0), sum(y == 1)]), np.argmax([
            sum(y == 0), sum(y == 1)])
        num_up = np.abs(sum(y == 0) - sum(y == 1))
        if sel.how_resample == 'up_resample':
            y_need_up = y[y == ind_up]
            x_need_up = x[y == ind_up]

            x = np.vstack([x[y == ind_down], x_need_up[:num_up, :], x_need_up])

            # dropna
            x = pd.DataFrame(x)
            x = x.dropna()
            ind = list(x.index)
            x = x.values

            y = np.hstack([y[y == ind_down], y_need_up[:num_up], y_need_up])
            y = pd.DataFrame(y).loc[ind].values

        else:
            y_need_down = y[y == ind_down]
            x_need_down = x[y == ind_down]

            # dropna
            x = np.vstack([x_need_down[num_up:, :], x[y == ind_up]])
            x = pd.DataFrame(x)
            x = x.dropna()
            ind = list(x.index)
            x = x.values

            y = np.hstack([y_need_down[num_up:], y[y == ind_up]])
            y = pd.DataFrame(y).loc[ind].values

        print('平衡后的样本={}:{}\n'.format(sum(y == 0), sum(y == 1)))
        sel.balanced_sample_size = '{}:{}'.format(sum(y == 0), sum(y == 1))

        # 置换y
#        rand_ind=np.random.permutation(len(y))
#        y=y[rand_ind]

        # cross-validation
        # 1) split data to training and testing datasets
#        x_train, x_test, y_train, y_test = \
#                            train_test_split(x, y, random_state=sel.random_state)

        # rfe
        import lc_svc_rfe_cv_V2 as lsvc
        model = lsvc.svc_rfe_cv(k=sel.k, pca_n_component=0.85)

        results = model.main_svc_rfe_cv(x, y)

        results = results.__dict__

        return results


if __name__ == '__main__':
    import lc_classify_FC as Clasf
    sel = Clasf.classify_using_FC(
        k=5,
        file_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\x_test206',
        dataset_name=None,
        scale=r'D:\WorkStation_2018\WorkStation_dynamicFC\Scales\8.30大表.xlsx',
        save_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic',
        feature='static',
        folder_label_name='folder_label_static_add.xlsx',
        which_group_to_classify=[1, 3],
        how_resample='down_resample',
        mask=np.ones([114, 114]),
        n_processess=10,
        if_save_post_mat=1,
        random_state=2)

    results = sel.load_allmat()
    results = sel.gen_label()
    result = sel.machine_learning()
