# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 21:40:21 2018

@author: lenovo
"""


import multiprocessing
import time
#from scipy import io
import sys  
sys.path.append(r'D:\myCodes\LC_MVPA\Python\MVPA_Python\utils')
# import module
from lc_read_write_Mat import write_mat
import time,os
import numpy as np
import lc_svc_rfe_cv as lsvc


##
class Perm_mvpa():
#    # initial parameters
    def __init__(self,\
                 model=lsvc.svc_rfe_cv(permutation=1,num_jobs=1),\
                 N_perm=20,\
                 batchsize=100,\
                 n_processess=5,\
                 fileName=r'D:\myCodes\LC_MVPA\Python\MVPA_Python\perm',\
                 k=5):
        self.model=model
        self.N_perm=N_perm
        self.batchsize=batchsize
        self.n_processess=n_processess
        self.fileName=fileName
        self.k=k # k fold CV of model
        
##
    def perm_mvpa(self,X,y):
        s=time.time()
        blocks=int(np.ceil(self.N_perm/self.batchsize))
        s=time.time()
        start=0
        end=min(self.batchsize,self.N_perm)# make sure no greater than N_erm
        for i in range(blocks):
            print('running {}/{}......'.format(i+1,blocks))
            pool = multiprocessing.Pool(processes=self.n_processess)
            for n_perm in np.arange(start,end):
                pool.apply_async(self.run_svc,\
                (X,y,n_perm))
            start+=self.batchsize
            end=min(end+self.batchsize,self.N_perm)
            pool.close()   
            pool.join()
        e=time.time()
        print('Done!\n running time is {:.1f}'.format(e-s))
    
    
    #    
    def run_svc(self,X,y,n_perm):
#        print('we have processing {} permutation'.format(n_perm))
        y_rand=np.random.permutation(y)
        predict,dec,y_sorted,weight=\
        self.model.main_svc_rfe_cv(X,y_rand,self.k)
    
    #     write mat
        write_mat(os.path.join(self.fileName,str(n_perm)),\
                  dataset_name=['predict','dec','y_sorted','weight'],\
                 dataset=[predict,dec,y_sorted,weight])
###
        

if __name__=='__main__':
    import lc_permutation_svc_multiprocessing_block as Perm
    perm=Perm.Perm_mvpa()
    perm.perm_mvpa(X,y)
#    perm.run_svc(X,y,1)
