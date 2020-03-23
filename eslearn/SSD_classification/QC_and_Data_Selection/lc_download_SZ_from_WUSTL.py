# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:07:17 2019
This script is used to download resting state fmri data of schizophrenia patients and controls from WUSTL.
@author: lenovo
"""
import os
import numpy as np
import pandas as pd
from urllib import request
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# Inputs
participant_id_file = r'H:\Data\精神分裂症\ds000030\schizophrenia_WUSTL_taskfmri\participants.tsv'
n_processess = 8  
save_path = r'H:\Data\精神分裂症\ds000030\schizophrenia_WUSTL_taskfmri'

# Read participants id and selecte SZ and HC
subid = pd.read_csv(participant_id_file, sep='\t')
subid = subid['participant_id'][subid['condit'].isin(['SCZ','CON'])]

# Identify all nii.gz files and json files in website
path_img = ['https://openneuro.org/crn/datasets/ds000115/snapshots/00001/files/' + sid + ':func:' + sid + '_task-letter0backtask_bold.nii.gz' for sid in subid]
# path_json = ['https://openneuro.org/crn/datasets/ds000115/snapshots/00001/files/' + sid + ':func:' + sid + '_task-letter0backtask_bold.json' for sid in subid]

# Save all files website links to local directory
# pd.DataFrame(path_img).to_csv(r'H:\Data\精神分裂症\ds000030\SZimgpath.txt',index=False, header=False)
# pd.DataFrame(path_json).to_csv(r'H:\Data\精神分裂症\ds000030\SZjsonpath.txt',index=False, header=False)

# Downloading
def download(ith, img, sid):
    print(f'{ith+1}/{nfile}\n')
    save_imgpath = os.path.join(save_path, sid + '.nii.gz')
    if not os.path.exists(save_imgpath):
        request.urlretrieve(img,save_imgpath)
    elif (os.path.exists(save_imgpath)) and (os.path.getsize(save_imgpath) < 16727328):
        request.urlretrieve(img,save_imgpath)

nfile = len(path_img)
cores = multiprocessing.cpu_count()
if n_processess > cores:
    n_processess = cores - 1

with ThreadPoolExecutor(n_processess) as executor:
    for ith, (img, sid) in enumerate(zip(path_img, subid)):
        executor.submit(download, ith, img, sid)
        
print('Done!\n')