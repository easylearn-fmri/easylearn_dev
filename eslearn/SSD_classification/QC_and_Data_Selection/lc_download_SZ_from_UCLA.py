# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:07:17 2019
This script is used to download resting state fmri data of schizophrenia patients and controls from ds000030 (UCLA dataset).
@author: lenovo
"""
import os
import numpy as np
import pandas as pd
from urllib import request
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# Inputs
participant_id_file = r'H:\Data\精神分裂症\ds000030\schizophrenia_UCLA_restfmri\participants.tsv'
n_processess = 8  
save_path = r'H:\Data\精神分裂症\ds000030\schizophrenia_UCLA_restfmri'

# Read participants id and selecte SZ and HC
subid = pd.read_csv(participant_id_file,sep='\t')
subid = subid['participant_id'][subid['diagnosis'].isin(['SCHZ','CONTROL'])][subid['rest']==1]

# Identify all nii.gz files and json files in website
path_img = ['https://openneuro.org/crn/datasets/ds000030/snapshots/00016/files/' + sid + ':func:' + sid + '_task-rest_bold.nii.gz' for sid in subid]
path_json = ['https://openneuro.org/crn/datasets/ds000030/snapshots/00016/files/' + sid + ':func:' + sid + '_task-rest_bold.json' for sid in subid]

# Save all files website links to local directory
pd.DataFrame(path_img).to_csv(os.path.join(save_path,'SZimgpath.txt'),index=False, header=False)
pd.DataFrame(path_img).to_csv(os.path.join(save_path,'SZjsonpath.txt'),index=False, header=False)
# Downloading
def download(ith, img, json, sid):
    print(f'{ith+1}/{nfile}\n')
    save_imgpath = os.path.join(save_path, sid + '.nii.gz')
    save_jsonpath = os.path.join(save_path, sid + '.json')
    if not os.path.exists(save_imgpath):
        request.urlretrieve(img,save_imgpath)
    elif (os.path.exists(save_imgpath)) and (os.path.getsize(save_imgpath) < 22728178):
        request.urlretrieve(img,save_imgpath)

        
    if not os.path.exists(save_jsonpath):
        request.urlretrieve(json,save_jsonpath)

nfile = len(path_img)
cores = multiprocessing.cpu_count()
if n_processess > cores:
    n_processess = cores - 1

with ThreadPoolExecutor(n_processess) as executor:
    for ith, (img, json, sid) in enumerate(zip(path_img, path_json, subid)):
        executor.submit(download, ith, img, json, sid)
        
print('Done!\n')