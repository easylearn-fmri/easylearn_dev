"""
This script is designed to perform table statistics
"""

import pandas as pd
import numpy as np
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import os
from Utils.lc_read_write_mat import read_mat

#%% ----------------------------------Our center 550----------------------------------
uid_path_550 = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\selected_550.txt'
scale_path_550 = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\10-24大表.xlsx'

scale_data_550 = pd.read_excel(scale_path_550)
uid_550 = pd.read_csv(uid_path_550, header=None)

scale_selected_550 = pd.merge(uid_550, scale_data_550, left_on=0, right_on='folder', how='inner')
describe_bprs_550 = scale_selected_550.groupby('诊断')['BPRS_Total'].describe()
describe_age_550 = scale_selected_550.groupby('诊断')['年龄'].describe()
describe_duration_550 = scale_selected_550.groupby('诊断')['病程月'].describe()
describe_durgnaive_550 = scale_selected_550.groupby('诊断')['用药'].value_counts()
describe_sex_550 = scale_selected_550.groupby('诊断')['性别'].value_counts()

#%% ----------------------------------BeiJing 206----------------------------------
uid_path_206 = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\北大精分人口学及其它资料\SZ_NC_108_100.xlsx'
scale_path_206 = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\北大精分人口学及其它资料\SZ_NC_108_100-WF.csv'
uid_to_remove = ['SZ010109','SZ010009']

scale_data_206 = pd.read_csv(scale_path_206)
scale_data_206 = scale_data_206.drop(np.array(scale_data_206.index)[scale_data_206['ID'].isin(uid_to_remove)])
scale_data_206['PANSStotal1'] = np.array([np.float64(duration) if duration.strip() !='' else 0 for duration in scale_data_206['PANSStotal1'].values])
Pscore = pd.DataFrame(scale_data_206[['P1', 'P2', 'P3', 'P4', 'P4', 'P5', 'P6', 'P7']].iloc[:106,:], dtype = np.float64)

Pscore = np.sum(Pscore, axis=1).describe()
Nscore = pd.DataFrame(scale_data_206[['N1', 'N2', 'N3', 'N4', 'N4', 'N5', 'N6', 'N7']].iloc[:106,:], dtype=np.float64)
Nscore = np.sum(Nscore, axis=1).describe()

Gscore = pd.DataFrame(scale_data_206[['G1', 'G2', 'G3', 'G4', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16']].iloc[:106,:])
Gscore = np.array(Gscore)
for i, itemi in enumerate(Gscore):
    for j, itemj in enumerate(itemi):
        print(itemj)
        if itemj.strip() != '':
            Gscore[i,j] = np.float64(itemj)
        else:
            Gscore[i, j] = np.nan
Gscore = pd.DataFrame(Gscore)      
Gscore = np.sum(Gscore, axis=1).describe()

describe_panasstotol_206 = scale_data_206.groupby('group')['PANSStotal1'].describe()
describe_age_206 = scale_data_206.groupby('group')['age'].describe()
scale_data_206['duration'] = np.array([np.float64(duration) if duration.strip() !='' else 0 for duration in scale_data_206['duration'].values])
describe_duration_206 = scale_data_206.groupby('group')['duration'].describe()
describe_sex_206 = scale_data_206.groupby('group')['sex'].value_counts()

#%% -------------------------COBRE----------------------------------
# Inputs
matroot = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\SelectedFC_COBRE'  # all mat files directory
scale = r'H:\Data\精神分裂症\COBRE\COBRE_phenotypic_data.csv'  # whole scale path

# Transform the .mat files to one .npy file
allmatname = os.listdir(matroot)

# Give labels to each subject, concatenate at the first column
allmatname = pd.DataFrame(allmatname)
allsubjname = allmatname.iloc[:,0].str.findall(r'[1-9]\d*')
allsubjname = pd.DataFrame([name[0] for name in allsubjname])
scale_data = pd.read_csv(scale,sep=',',dtype='str')
print(scale_data)
diagnosis = pd.merge(allsubjname,scale_data,left_on=0,right_on='ID')[['ID','Subject Type']]
scale_data = pd.merge(allsubjname,scale_data,left_on=0,right_on='ID')

diagnosis['Subject Type'][diagnosis['Subject Type'] == 'Control'] = 0
diagnosis['Subject Type'][diagnosis['Subject Type'] == 'Patient'] = 1
include_loc = diagnosis['Subject Type'] != 'Disenrolled'
diagnosis = diagnosis[include_loc.values]
allsubjname = allsubjname[include_loc.values]
scale_data_COBRE = pd.merge(allsubjname, scale_data, left_on=0, right_on=0, how='inner').iloc[:,[0,1,2,3,5]]
scale_data_COBRE['Gender'] = scale_data_COBRE['Gender'].str.replace('Female', '0')
scale_data_COBRE['Gender'] = scale_data_COBRE['Gender'].str.replace('Male', '1')
scale_data_COBRE['Subject Type'] = scale_data_COBRE['Subject Type'].str.replace('Patient', '1')
scale_data_COBRE['Subject Type'] = scale_data_COBRE['Subject Type'].str.replace('Control', '0')
scale_data_COBRE = pd.DataFrame(scale_data_COBRE, dtype=np.float64)

describe_age_COBRE = scale_data_COBRE.groupby('Subject Type')['Current Age'].describe()
describe_sex_COBRE = scale_data_COBRE.groupby('Subject Type')['Gender'].value_counts()


#%% -------------------------UCLA----------------------------------
matroot = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\SelectedFC_UCLA'
scale = r'H:\Data\精神分裂症\ds000030\schizophrenia_UCLA_restfmri\participants.tsv'

allmatname = os.listdir(matroot)
allmatname = pd.DataFrame(allmatname)
allsubjname = allmatname.iloc[:,0].str.findall(r'[1-9]\d*')
allsubjname = pd.DataFrame(['sub-' + name[0] for name in allsubjname])
scale_data = pd.read_csv(scale,sep='\t')
scale_data_UCAL = pd.merge(allsubjname,scale_data,left_on=0,right_on='participant_id')
scale_data_UCAL['diagnosis'][scale_data_UCAL['diagnosis'] == 'CONTROL']=0
scale_data_UCAL['diagnosis'][scale_data_UCAL['diagnosis'] == 'SCHZ']=1
scale_data_UCAL['participant_id'] = scale_data_UCAL['participant_id'].str.replace('sub-', '')
scale_data_UCAL = pd.merge(allsubjname,scale_data_UCAL, left_on=0, right_on=0, how='inner')
scale_data_UCAL = scale_data_UCAL.iloc[:,[2,3,4]]
scale_data_UCAL['gender'] = scale_data_UCAL['gender'].str.replace('m', '1')
scale_data_UCAL['gender'] = scale_data_UCAL['gender'].str.replace('f', '0')
scale_data_UCAL = pd.DataFrame(scale_data_UCAL, dtype=np.float64)
describe_age_UCAL = scale_data_UCAL.groupby('diagnosis')['age'].describe()
describe_sex_UCAL = scale_data_UCAL.groupby('diagnosis')['gender'].value_counts()
#%%--------------------------------------------------------------------