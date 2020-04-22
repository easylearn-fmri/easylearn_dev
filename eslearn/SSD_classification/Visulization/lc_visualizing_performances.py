# -*- coding: utf-8 -*-
"""
This script is used to perform post-hoc analysis and visualization: 
the classification performance of subsets (only for Schizophrenia Spectrum: SZ and Schizophreniform).
Unless otherwise specified, all results  are for Schizophrenia Spectrum.
"""

#%%
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
sys.path.append(r'D:\My_Codes\easylearn-fmri\eslearn\statistical_analysis')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
import pickle

from lc_binomialtest import lc_binomialtest
from eslearn.statistical_analysis.lc_anova import oneway_anova
from eslearn.visualization.el_violine import ViolinPlot

#%% Inputs
scale_550_file = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'
scale_206_file = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\SZ_NC_108_100-WF.csv'
scale_206_drug_file = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\SZ_109_drug.xlsx'
headmotion_file_dataset1 = r'D:\WorkStation_2018\SZ_classification\Scale\头动参数_1322.xlsx'
classification_results_pooling_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_pooling.npy'
classification_results_results_leave_one_site_cv_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_leave_one_site_cv.npy'
classification_results_feu_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_unmedicated_and_firstepisode_550.npy'
is_plot = 1
is_savefig = 0

#%% Load and proprocess
scale_550 = pd.read_excel(scale_550_file)
headmotion_dataset1 = pd.read_excel(headmotion_file_dataset1)[['Subject ID', 'mean FD_Power']]
scale_550 = pd.merge(scale_550, headmotion_dataset1, left_on='folder', right_on='Subject ID', how='inner')

scale_206 = pd.read_csv(scale_206_file)
scale_206_drug = pd.read_excel(scale_206_drug_file)

results_pooling = np.load(classification_results_pooling_file, allow_pickle=True)
results_leave_one_site_cv = np.load(classification_results_results_leave_one_site_cv_file, allow_pickle=True)
results_feu = np.load(classification_results_feu_file, allow_pickle=True)

results_special = results_pooling['special_result']
results_special = pd.DataFrame(results_special)
results_special.iloc[:, 0] = np.int32(results_special.iloc[:, 0])

scale_206['ID'] = scale_206['ID'].str.replace('NC','10')
scale_206['ID'] = scale_206['ID'].str.replace('SZ','20')
scale_206['ID'] = np.int32(scale_206['ID'])
scale_550['folder'] = np.int32(scale_550['folder'])

scale_206_drug['P0001'] = scale_206_drug['P0001'].str.replace('NC','10')
scale_206_drug['P0001'] = scale_206_drug['P0001'].str.replace('SZ','20')
scale_206_drug['P0001'] = np.int32(scale_206_drug['P0001'])

# Filter subjects that have .mat files
scale_550_selected = pd.merge(results_special, scale_550, left_on=0, right_on='folder', how='inner')
scale_206_selected = pd.merge(results_special, scale_206, left_on=0, right_on='ID', how='inner')
scale_206_selected = pd.merge(scale_206_selected, scale_206_drug, left_on=0, right_on='P0001', how='inner')

#%% ---------------------------------Calculate performance for Schizophrenia Spectrum subgroups-------------------------------
## Step 1: Dataset1
duration = 18  # Upper limit of first episode: 
""" reference:
1. Kane JM, Robinson DG, Schooler NR, et al. Comprehensive versus usual
community care for first-episode psychosis: 2-year outcomes from the NIMH
RAISE early treatment program. Am J Psychiatry. 2016;173(4):362-372. doi:10.1176/appi.ajp.2015.15050632.
2. Cognitive Impairment in Never-Medicated Individuals on the Schizophrenia Spectrum. doi:10.1001/jamapsychiatry.2020.0001"
"""

data_firstepisode_SZ_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['首发'] == 1) & (scale_550_selected['病程月'] <= duration) & (scale_550_selected['病程月'] >= 6)]
data_not_firstepisode_SZ_550 = scale_550_selected[(scale_550_selected['诊断']==3) & ((scale_550_selected['首发'] == 0) | (scale_550_selected['病程月'] > duration))]  # Including the persistent patients

data_schizophreniform_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['病程月'] < 6)]
data_shortdurationSZ_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['病程月'] <= duration) & (scale_550_selected['病程月'] >= 6)]
data_longdurationSZ_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['病程月'] > duration)]

onsetage_all_550 = scale_550_selected['Age_of_first_episode'][scale_550_selected['诊断']==3].dropna()
ind_young_onsetage_550 = onsetage_all_550.index[onsetage_all_550.values <= np.median(onsetage_all_550)]
ind_old_onsetage_550 = onsetage_all_550.index[onsetage_all_550.values > np.median(onsetage_all_550)]
data_young_onset_age_550 = scale_550_selected[scale_550_selected['诊断']==3].loc[ind_young_onsetage_550]
data_old_onset_age_550 = scale_550_selected[scale_550_selected['诊断']==3].loc[ind_old_onsetage_550]

data_medicated_SSD_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['用药'] == 1)]
data_unmedicated_SSD_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['用药'] == 0) ]

# Frist episode and unmedicated
data_unmedicated_schizophreniform_550 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['病程月'] < 6) & 
    (scale_550_selected['用药'] == 0)
]

data_unmedicated_SZ_550 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['病程月'] >= 6) & 
    (scale_550_selected['用药'] == 0)
]

data_firstepisode_unmedicated_SZ_550 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['首发'] == 1) &
    (scale_550_selected['病程月'] <= duration) & 
    (scale_550_selected['病程月'] >= 6) & 
    (scale_550_selected['用药'] == 0)
]

data_chronic_unmedicated_SZ_550 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['病程月'] > duration) & 
    (scale_550_selected['用药'] == 0)
]

# data_unmedicated_SSD_550['folder'].to_csv(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\feu_63.txt', index=False)

## Calculating Accuracy
acc_firstepisode_SZ_550 = np.sum(data_firstepisode_SZ_550[1]-data_firstepisode_SZ_550[3]==0)/len(data_firstepisode_SZ_550)
acc_not_firstepisode_SZ_550 = np.sum(data_not_firstepisode_SZ_550[1]-data_not_firstepisode_SZ_550[3]==0)/len(data_not_firstepisode_SZ_550)

acc_schizophreniform_550 = np.sum(data_schizophreniform_550[1]-data_schizophreniform_550[3]==0)/len(data_schizophreniform_550)
acc_shortduration_550 = np.sum(data_shortdurationSZ_550[1]-data_shortdurationSZ_550[3]==0)/len(data_shortdurationSZ_550)
acc_longduration_550 = np.sum(data_longdurationSZ_550[1]-data_longdurationSZ_550[3]==0)/len(data_longdurationSZ_550)

acc_young_onsetage_550 = np.sum(data_young_onset_age_550[1]-data_young_onset_age_550[3]==0)/len(data_young_onset_age_550)
acc_old_onsetage_550 = np.sum(data_old_onset_age_550[1]-data_old_onset_age_550[3]==0)/len(data_old_onset_age_550)

acc_medicated_SSD_550 = np.sum(data_medicated_SSD_550[1]-data_medicated_SSD_550[3]==0)/len(data_medicated_SSD_550)
acc_ummedicated_SSD_550 = np.sum(data_unmedicated_SSD_550[1]-data_unmedicated_SSD_550[3]==0)/len(data_unmedicated_SSD_550)

acc_unmedicated_schizophreniform_550 = np.sum(data_unmedicated_schizophreniform_550[1]-data_unmedicated_schizophreniform_550[3]==0) / len(data_unmedicated_schizophreniform_550)
acc_unmedicated_SZ_550 = np.sum(data_unmedicated_SZ_550[1]-data_unmedicated_SZ_550[3]==0) / len(data_unmedicated_SZ_550)
acc_firstepisode_unmedicated_SZ_550 = np.sum(data_firstepisode_unmedicated_SZ_550[1]-data_firstepisode_unmedicated_SZ_550[3]==0) / len(data_firstepisode_unmedicated_SZ_550)
acc_chronic_unmedicated_SZ_550 = np.sum(data_chronic_unmedicated_SZ_550[1]-data_chronic_unmedicated_SZ_550[3]==0) / len(data_chronic_unmedicated_SZ_550)

print(f'Accuracy of firstepisode in dataset550 = {acc_firstepisode_SZ_550}')
print(f'Accuracy of none-firstepisode in dataset550 = {acc_not_firstepisode_SZ_550}')

print(f'Accuracy of schizophreniform in dataset550 = {acc_schizophreniform_550}')
print(f'Accuracy of shortduration in dataset550 = {acc_shortduration_550}')
print(f'Accuracy of longduration in dataset550 = {acc_longduration_550}')

print(f'Accuracy of young onsetage of 550 = {acc_young_onsetage_550}')
print(f'Accuracy of old onsetage of 550 = {acc_old_onsetage_550}')

print(f'Accuracy of medicated SSD in dataset550 = {acc_medicated_SSD_550}')
print(f'Accuracy of ummedicated_SSD in dataset550 = {acc_ummedicated_SSD_550}')

print(f'Accuracy of firstepisode unmedicated SZ in dataset550 = {acc_firstepisode_unmedicated_SZ_550}')
print('-'*50)

# Step 2: Dataset 2
## Preprocessing
scale_206_selected['duration'] = [np.int32(duration) if duration != ' ' else 10000 for duration in scale_206_selected['duration']]
scale_206_selected['firstepisode'] = [np.int32(firstepisode) if firstepisode != ' ' else 10000 for firstepisode in scale_206_selected['firstepisode']]
scale_206_selected['CPZ_eq'] = [np.int32(duration) if duration != ' ' else 0 for duration in scale_206_selected['CPZ_eq']]
scale_206_selected['onsetage'] = [np.int32(duration) if duration != ' ' else 0 for duration in scale_206_selected['onsetage']]

## Filter subgroups
data_firstepisode_206 = scale_206_selected[(scale_206_selected['group']==1) & (scale_206_selected['firstepisode'] == 1) & (scale_206_selected['duration'] <= duration)]
data_notfirstepisode_206 = scale_206_selected[(scale_206_selected['group']==1) & ((scale_206_selected['firstepisode'] == 0) | (scale_206_selected['duration'] > duration))]

data_shortduration_206 = scale_206_selected[(scale_206_selected['group']==1) & (scale_206_selected['duration'] <= duration)]
data_longduration_206 = scale_206_selected[(scale_206_selected['group']==1) & (scale_206_selected['duration'] > duration)]

onsetage = scale_206_selected['onsetage'][scale_206_selected['group']==1]
data_young_onsetage_206 = scale_206_selected[(scale_206_selected['group']==1) & (onsetage <= np.median(onsetage))]
data_old_onsetage_206 = scale_206_selected[(scale_206_selected['group']==1) & (onsetage > np.median(onsetage))]

CPZ_eq = scale_206_selected['CPZ_eq'][scale_206_selected['group']==1]
data_drugless_206 = scale_206_selected[(scale_206_selected['group']==1) & (CPZ_eq <= np.median(CPZ_eq))]
data_drugmore_206 = scale_206_selected[(scale_206_selected['group']==1) & (CPZ_eq > np.median(CPZ_eq))]

## Calculating acc
acc_firstepisode_206 = np.sum(data_firstepisode_206[1]-data_firstepisode_206[3]==0)/len(data_firstepisode_206)
acc_notfirstepisode_206 = np.sum(data_notfirstepisode_206[1]-data_notfirstepisode_206[3]==0)/len(data_notfirstepisode_206)

acc_shortduration_206 = np.sum(data_shortduration_206[1]-data_shortduration_206[3]==0)/len(data_shortduration_206)
acc_longduration_206 = np.sum(data_longduration_206[1]-data_longduration_206[3]==0)/len(data_longduration_206)

acc_young_onsetage_206 = np.sum(data_young_onsetage_206[1]-data_young_onsetage_206[3]==0)/len(data_young_onsetage_206)
acc_old_onsetage_206 = np.sum(data_old_onsetage_206[1]-data_old_onsetage_206[3]==0)/len(data_old_onsetage_206)

acc_drugless_206 = np.sum(data_drugless_206[1]-data_drugless_206[3]==0)/len(data_drugless_206)
acc_drugmore_206 = np.sum(data_drugmore_206[1]-data_drugmore_206[3]==0)/len(data_drugmore_206)

##
print(f'Accuracy of first episode of 206 = {acc_firstepisode_206}')
print(f'Accuracy of recurrent of 206 = {acc_notfirstepisode_206}')

print(f'Accuracy of shortduration of 206 = {acc_shortduration_206}')
print(f'Accuracy of longduration of 206 = {acc_longduration_206}')

print(f'Accuracy of young onsetage of 206 = {acc_young_onsetage_206}')
print(f'Accuracy of old onsetage of 206 = {acc_old_onsetage_206}')

print(f'Accuracy of drugless of 206 = {acc_drugless_206}')
print(f'Accuracy of drugmore of 206 = {acc_drugmore_206}')

#%% -------------------------------------Visualization-----------------------------------------------
if is_plot:
    accuracy_pooling = results_pooling['accuracy']
    sensitivity_pooling = results_pooling['sensitivity']
    specificity_pooling = results_pooling['specificity']
    AUC_pooling = results_pooling['AUC']
    performances_pooling = [accuracy_pooling, sensitivity_pooling, specificity_pooling, AUC_pooling]
    performances_pooling = pd.DataFrame(performances_pooling)

    accuracy_leave_one_site_cv = results_leave_one_site_cv['accuracy']
    sensitivity_leave_one_site_cv = results_leave_one_site_cv['sensitivity']
    specificity_leave_one_site_cv = results_leave_one_site_cv['specificity']
    AUC_leave_one_site_cv = results_leave_one_site_cv['AUC']
    performances_leave_one_site_cv = [accuracy_leave_one_site_cv, sensitivity_leave_one_site_cv, specificity_leave_one_site_cv, AUC_leave_one_site_cv]
    performances_leave_one_site_cv = pd.DataFrame(performances_leave_one_site_cv)

    accuracy_feu = results_feu['accuracy']
    sensitivity_feu = results_feu['sensitivity']
    specificity_feu = results_feu['specificity']
    AUC_feu = results_feu['AUC']
    performances_feu = [accuracy_feu, sensitivity_feu, specificity_feu, AUC_feu]
    performances_feu = pd.DataFrame(performances_feu)
    
    # Save weights to .mat file for visulazation using MATLAB.
    import scipy.io as io
    weight_pooling_1d = np.mean(np.squeeze(np.array(results_pooling['coef'])), axis=0)
    weight_leave_one_out_cv_1d = np.mean(np.squeeze(np.array(results_leave_one_site_cv['coef'])), axis=0)
    weight_feu_1d = np.mean(np.squeeze(np.array(results_feu['coef'])), axis=0)
    mask = np.triu(np.ones([246, 246]), 1) == 1
    weight_pooling = np.zeros([246, 246])
    weight_leave_one_out_cv = np.zeros([246, 246])
    weight_feu = np.zeros([246, 246])
    weight_pooling[mask] = weight_pooling_1d
    weight_leave_one_out_cv[mask] = weight_leave_one_out_cv_1d
    weight_pooling = weight_pooling + weight_pooling.T;
    weight_leave_one_out_cv = weight_leave_one_out_cv + weight_leave_one_out_cv.T
    weight_feu = weight_feu + weight_feu.T
    io.savemat(r'D:\WorkStation_2018\SZ_classification\Figure\weights.mat',
                {'weight_pooling': weight_pooling,
                'weight_leave_one_out_cv': weight_leave_one_out_cv,
                'weight_feu': weight_feu})


    # Bar: performances in the whole Dataset.
    import seaborn as sns
    plt.figure(figsize=(20,20))
    all_mean = np.concatenate([np.mean(performances_pooling.values,1), np.mean(performances_leave_one_site_cv.values,1), np.mean(performances_feu.values,1)])
    error = np.concatenate([np.std(performances_pooling.values, 1), np.std(performances_leave_one_site_cv.values, 1), np.std(performances_feu.values, 1)])

    plt.subplot(2, 1, 1)
    color = ['darkturquoise'] * 4 +  ['paleturquoise'] * 4 + ['lightblue'] * 4
    plt.bar(np.arange(0,len(all_mean)), all_mean, yerr = error, 
            capsize=5, linewidth=2, color=color)
    plt.tick_params(labelsize=20)
    plt.xticks(np.arange(0,len(all_mean)), ['Accuracy', 'Sensitivity', 'Sensitivity', 'AUC'] * 3, fontsize=20, rotation=45)
    plt.title('Classification performances', fontsize=25, fontweight='bold')
    y_major_locator=MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='y')
    plt.fill_between(np.linspace(-0.4,3.4), 0.95, 1.08, color='darkturquoise')
    plt.fill_between(np.linspace(3.6, 7.4), 0.95, 1.08, color='paleturquoise')
    plt.fill_between(np.linspace(7.6, 11.4), 0.95, 1.08, color='lightblue')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

                   
    # Bar: Dataset 1
    plt.subplot(2,1,2)
    barcont_550 = [
        acc_firstepisode_SZ_550, acc_not_firstepisode_SZ_550,
        acc_schizophreniform_550, acc_shortduration_550, acc_longduration_550, 
        acc_young_onsetage_550, acc_old_onsetage_550, 
        acc_medicated_SSD_550, acc_ummedicated_SSD_550, acc_unmedicated_schizophreniform_550, 
        acc_unmedicated_SZ_550, acc_firstepisode_unmedicated_SZ_550, acc_chronic_unmedicated_SZ_550
    ]

    label_550 = ['First episode SZ', 'Recurrent SZ', 'Schizophreniform', 'Short duration SZ', 'Long duration SZ',
                'Young onset age SSD','Elder onset age SSD', 
                'Medicated SSD', 'Unmedicated SSD', 
                'Unmedicated schizophreniform', 'Unmedicated SZ', 'First episode unmedicated SZ', 'Recurrent unmedicated SZ']

    samplesize_550 = [
        data_firstepisode_SZ_550.shape[0], data_not_firstepisode_SZ_550.shape[0],
        data_schizophreniform_550.shape[0], data_shortdurationSZ_550.shape[0], data_longdurationSZ_550.shape[0], 
        data_young_onset_age_550.shape[0], data_old_onset_age_550.shape[0], 
        data_medicated_SSD_550.shape[0], data_unmedicated_SSD_550.shape[0], data_unmedicated_schizophreniform_550.shape[0], 
        data_unmedicated_SZ_550.shape[0], data_firstepisode_unmedicated_SZ_550.shape[0], data_chronic_unmedicated_SZ_550.shape[0]
    ]


    mean_550 = [
        0, 0, 0, 
        data_shortdurationSZ_550['病程月'].mean(), data_longdurationSZ_550['病程月'].mean(), 
        data_young_onset_age_550['Age_of_first_episode'].mean(), data_old_onset_age_550['Age_of_first_episode'].mean(), 
         0, 0, 0, 
        0, 0, 0
    ]

    std_550 = [
        0, 0, 0, 
        data_shortdurationSZ_550['病程月'].std(), data_longdurationSZ_550['病程月'].std(), 
        data_young_onset_age_550['Age_of_first_episode'].std(), data_old_onset_age_550['Age_of_first_episode'].std(), 
         0, 0, 0, 
        0, 0, 0
    ] 


    # Bar: Dataset 2
    barcont_206 = [
        acc_firstepisode_206, acc_notfirstepisode_206,
        acc_shortduration_206, acc_longduration_206, 
        acc_young_onsetage_206, acc_old_onsetage_206,
        acc_drugless_206, acc_drugmore_206
    ]

    label_206 = [
        'First episode SZ', 'Recurrent SZ', 'Short duration SZ', 'Long duration SZ',
        'Young onset age SZ','Elder onset age SZ', 
        'Larger dosage SZ', 'Small dosage SZ'
    ]

    samplesize_206 = [
        data_firstepisode_206.shape[0], data_notfirstepisode_206.shape[0],
        data_shortduration_206.shape[0], data_longduration_206.shape[0], 
        data_young_onsetage_206.shape[0], data_old_onsetage_206.shape[0],
        data_drugless_206.shape[0], data_drugmore_206.shape[0]
    ]     

    mean_206 = [
        0, 0,
        data_shortduration_206['duration'].mean(), data_longduration_206['duration'].mean(), 
        data_young_onsetage_206['onsetage'].mean(), data_old_onsetage_206['onsetage'].mean(),
        data_drugless_206['CPZ_eq'].mean(), data_drugmore_206['CPZ_eq'].mean()
    ] 
    
    std_206 = [
        0, 0,
        data_shortduration_206['duration'].std(), data_longduration_206['duration'].std(), 
        data_young_onsetage_206['onsetage'].std(), data_old_onsetage_206['onsetage'].std(),
        data_drugless_206['CPZ_eq'].std(), data_drugmore_206['CPZ_eq'].std()
    ]

    ## Plot
    barcont_all = barcont_206 + barcont_550
    label_all = label_206 + label_550 
    mean_all = mean_206 + mean_550
    std_all = std_206 + std_550
    samplesize_all = samplesize_206 + samplesize_550
    color_206 = ['lightblue' for i in range(len(label_206))] 
    color_550 = ['darkturquoise' for i in range(len(label_550))]
    # h = plt.barh(np.arange(0, len(barcont_all)), barcont_all, color=color)
    
    h206 = plt.barh(np.arange(0, len(barcont_206)), barcont_206, color=color_206)
    h550 = plt.barh(np.arange(len(barcont_206), len(barcont_206) + len(barcont_550)), barcont_550, color=color_550)
    plt.legend([h550, h206], ['Dataset 1', 'Dataset 2'], fontsize=20)
    
    plt.tick_params(labelsize=20)
    plt.yticks(np.arange(0,len(barcont_all)), label_all, fontsize=20, rotation=0)
    plt.title('Sensitivity of each subgroup of SSD in dataset 1 and dateset 2', fontsize=25, fontweight='bold')

    x_major_locator=MultipleLocator(0.1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Sensitivity', fontsize=25) 
    plt.grid(axis='x')
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    for i, (y, x, n, m, s) in enumerate(zip(yticks, barcont_all, samplesize_all, mean_all, std_all)):
        p, _, _, _ = lc_binomialtest(n, np.int(n * x), 0.5, 0.5)
        
        if m: plt.text(0.101, y-0.3, f'mean={m:.1f}({s:.1f})', fontsize=15)
        plt.text(0.31, y-0.3, 'N = %.0f' % n, fontsize=15)
        if np.isin(i, (20, 19, 18, 17, 16)):
            plt.text(0.41, y-0.3, 'P = %.3f' % p, fontsize=16, color='k', fontweight='bold')
            plt.text(x+0.01, y-0.3, '%.2f' % x,fontsize=16, color='k', fontweight='bold')
        else:
            plt.text(0.41, y-0.3, 'P = %.3f' % p, fontsize=15)
            plt.text(x+0.01, y-0.3, '%.2f' % x,fontsize=15)

    #%% Save to PDF format
    if is_savefig & is_plot:
        plt.tight_layout()
        plt.subplots_adjust(left=0.25, wspace = 0.5, hspace = 0.5)  # wspace 左右
        pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Classification_performances_all_cutoff' + str(duration) + '.pdf')
        pdf.savefig()
        pdf.close()
    plt.show()

#%% Head motion
is_savefig_headmotion = 1

mean_meanFD_550 = [
    data_firstepisode_SZ_550['mean FD_Power'].mean(axis=0), data_not_firstepisode_SZ_550['mean FD_Power'].mean(axis=0),
    data_schizophreniform_550['mean FD_Power'].mean(axis=0), data_shortdurationSZ_550['mean FD_Power'].mean(axis=0), data_longdurationSZ_550['mean FD_Power'].mean(axis=0), 
    data_young_onset_age_550['mean FD_Power'].mean(axis=0), data_old_onset_age_550['mean FD_Power'].mean(axis=0), 
    data_medicated_SSD_550['mean FD_Power'].mean(axis=0), data_unmedicated_SSD_550['mean FD_Power'].mean(axis=0), data_unmedicated_schizophreniform_550['mean FD_Power'].mean(axis=0), 
    data_unmedicated_SZ_550['mean FD_Power'].mean(axis=0), data_firstepisode_unmedicated_SZ_550['mean FD_Power'].mean(axis=0), data_chronic_unmedicated_SZ_550['mean FD_Power'].mean(axis=0)
]

std_meanFD_550 = [
    data_firstepisode_SZ_550['mean FD_Power'].std(), data_not_firstepisode_SZ_550['mean FD_Power'].std(),
    data_schizophreniform_550['mean FD_Power'].std(), data_shortdurationSZ_550['mean FD_Power'].std(), data_longdurationSZ_550['mean FD_Power'].std(), 
    data_young_onset_age_550['mean FD_Power'].std(), data_old_onset_age_550['mean FD_Power'].std(), 
    data_medicated_SSD_550['mean FD_Power'].std(), data_unmedicated_SSD_550['mean FD_Power'].std(), data_unmedicated_schizophreniform_550['mean FD_Power'].std(), 
    data_unmedicated_SZ_550['mean FD_Power'].std(), data_firstepisode_unmedicated_SZ_550['mean FD_Power'].std(), data_chronic_unmedicated_SZ_550['mean FD_Power'].std()
]
  
meanFD_550 = [
    data_firstepisode_SZ_550['mean FD_Power'], data_not_firstepisode_SZ_550['mean FD_Power'],
    data_schizophreniform_550['mean FD_Power'], data_shortdurationSZ_550['mean FD_Power'], data_longdurationSZ_550['mean FD_Power'], 
    data_young_onset_age_550['mean FD_Power'], data_old_onset_age_550['mean FD_Power'], 
    data_medicated_SSD_550['mean FD_Power'], data_unmedicated_SSD_550['mean FD_Power'], data_unmedicated_schizophreniform_550['mean FD_Power'], 
    data_unmedicated_SZ_550['mean FD_Power'], data_firstepisode_unmedicated_SZ_550['mean FD_Power'], data_chronic_unmedicated_SZ_550['mean FD_Power']
]

f, p = oneway_anova(*meanFD_550)
ViolinPlot().plot(meanFD_550, ylabel='mean FD', ylabelsize=15, xticklabel=label_550, xticklabel_rotation=45)
y_major_locator=MultipleLocator(0.05)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

if is_savefig_headmotion:
    plt.tight_layout()
    plt.subplots_adjust(left=0.25, wspace = 0.5, hspace = 0.5)  # wspace 左右
    pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\headmotion_dataset1.pdf')
    pdf.savefig()
    pdf.close()
plt.show()
        

