# utf-8
"""
read .1D file
"""

import os
import pandas as pd
import numpy as np
from lc_read_write_Mat import write_mat


def read_1D(file_path):
	"""
	Transform the .1D file to .mat file
	TODO: to other file type
	
	Args:
		file_path: .1D file path
	Return:
		data: .mat file 
	"""
    data = pd.read_csv(file_path)
    data = data.values
    data = [list(d)[0].split('\t') for d in data]
    data =[np.array(d, dtype=np.float).T for d in data]
    data = pd.DataFrame(data)
    data = pd.DataFrame(data)
    return data
    

if __name__ == '__main__':
    dir = r'F:\Data\ASD\Outputs\dparsf\filt_global\rois_dosenbach160'
    save_path = r'F:\Data\ASD\Outputs\dparsf\filt_global\rois_dosenbach160_m'
    file_name = os.listdir(dir)
    file_path = [os.path.join(dir, fn) for fn in file_name]
    nfile = len(file_path)
    for i, file in enumerate(zip(file_path, file_name)):
        print(f'{i+1}/{nfile}')
        data = read_1D(file[0])
        fn = os.path.join(save_path, file[1].split('.')[0] + '.mat')
        write_mat(fileName=fn, dataset_name='timeseries', dataset=data.values)


