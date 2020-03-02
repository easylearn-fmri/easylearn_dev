# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:18:41 2018
permutation test: Parallel elastic net regression
Note.：The program is divided into many blocks
       so as to avoid interruption.
   input:
       fileName=fileName to save results
@author: Li Chao
"""
# import module
from joblib import Parallel, delayed
#from scipy import io
import sys  
sys.path.append(r'D:\myCodes\LC_MVPA\Python\MVPA_Python\utils')
from lc_write_read_h5py import write_h5py
#from read_write_Mat_LC import write_mat
import time
import numpy as np
import lc_elasticNet as enet



# def
def permutation(X,y,N_perm,batchsize,fileName):
    # instantiating object
    model=enet.elasticNet_LC(permutation=1,num_jobs=1)#
    blocks=int(np.ceil(N_perm/batchsize))
    start_time=time.clock()
    start=0
    end=batchsize
    for i in range(blocks):
        print('completed {:.1f}%'.format((i*100)/blocks))
        Parallel(n_jobs=8,backend='threading')\
        (delayed(run_enet)(X,y,n_perm,model,fileName)\
         for n_perm in np.arange(start,end))
        start+=batchsize
        end=min(end+batchsize,N_perm)
    end_time=time.clock()
    print('running time is: {:.1f} second'.format(end_time-start_time))  

def run_enet(X,y,n_perm,model,fileName):
    y_rand=np.random.permutation(y)
    Predict,y_sorted,Coef,r=model.elasticNetCV_Outer(X,y_rand)#
    # write h5py
    write_h5py(fileName,'perm'+str(n_perm),['Predict','y_sorted','Coef','r'],\
          [Predict,y_sorted,Coef,r])
    # write mat
#    write_mat(fileName='enet_test.mat',\
#              dataset_name=['Predict'+str(n_perm),\
#                            'y_sorted'+str(n_perm),\
#                            'Coef'+str(n_perm),\
#                            'r'+str(n_perm)],\
#             dataset=[Predict,y_sorted,Coef,r])
if __name__=='__main__':
    permutation(X,y,N_perm=5,batchsize=1,fileName='t4test')