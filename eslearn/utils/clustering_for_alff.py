# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:15:10 2018

@author: lenovo
"""

import lc_copy_selected_file_V6 as copy
import sys
import pandas as pd

# files

label_path = r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\new\label.xlsx'
scale_path = r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\new\大表.xlsx'

# load
scale = pd.read_excel(scale_path)
label = pd.read_excel(label_path)

# label folder
label_a = label['folder'][label['new'] == 'a']
label_c = label['folder'][label['new'] == 'c']
label_bde = pd.concat([
    label['folder'][label['new'] == 'b'],
    label['folder'][label['new'] == 'd'],
    label['folder'][label['new'] == 'e']
], axis=0)

# cov of label folder
cov_a = pd.DataFrame(label_a).merge(scale, on='folder', how='inner')
cov_a = cov_a[['folder', '年龄', '性别']].dropna()

cov_c = pd.DataFrame(label_c).merge(scale, on='folder', how='inner')
cov_c = cov_c[['folder', '年龄', '性别']].dropna()

cov_bde = pd.DataFrame(label_bde).merge(scale, on='folder', how='inner')
cov_bde = cov_bde[['folder', '年龄', '性别']].dropna()

hc = scale[scale['诊断'] == 1]
hc = hc[['folder', '年龄', '性别']]
hc = hc.iloc[:, [0, 1, 2]]
hc = hc.dropna()

# save folder and cov
cov_a['folder'].to_excel('folder_a.xlsx', index=False, header=False)
cov_c['folder'].to_excel('folder_c.xlsx', index=False, header=False)
cov_bde['folder'].to_excel('folder_bde.xlsx', index=False, header=False)

cov_a[['年龄', '性别']].to_csv('cov_a.txt', index=False, header=False, sep=' ')
cov_c[['年龄', '性别']].to_csv('cov_c.txt', index=False, header=False, sep=' ')
cov_bde[['年龄', '性别']].to_csv('cov_bde.txt', index=False, header=False, sep=' ')

# copy
sys.path.append(
    r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Utils')


path = r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\new'
folder = r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\new\folder_a.xlsx'
save_path = r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\new\ALFF_a'

sel = copy.copy_fmri(
    referencePath=folder,
    regularExpressionOfsubjName_forReference='([1-9]\d*)',
    ith_reference=0,
    folderNameContainingFile_forSelect='',
    num_countBackwards=1,
    regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',
    ith_subjName=0,
    keywordThatFileContain='^mALFF',
    neuroimageDataPath=path,
    savePath=save_path,
    n_processess=6,
    ifSaveLog=1,
    ifCopy=1,
    ifMove=0,
    saveInToOneOrMoreFolder='saveToOneFolder',
    saveNameSuffix='',
    ifRun=1)

result = sel.main_run()

results = result.__dict__
print(results.keys())
print('Done!')
