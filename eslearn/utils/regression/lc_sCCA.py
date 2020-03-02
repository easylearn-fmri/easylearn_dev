# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:41:55 2018
Please refer to and cite the follow paper:
{Pyrcca: regularized kernel canonical correlation analysis in 
python and its applications to neuroimaging}
@author: lenovo
"""

# search path append
import sys  
sys.path.append(r'D:\myCodes\LC_MVPA\Python\MVPA_Python\utils\regression\pyrcca-master\pyrcca-master')
# imports
import rcca,time,multiprocessing
#import pandas as pd
import numpy as np
#from sklearn.externals.joblib import Parallel, delayed

# Initialize number of samples
nSamples = 500

# Define two latent variables (number of samples x 1)
latvar1 = np.random.randn(nSamples,)
latvar2 = np.random.randn(nSamples,)

# Define independent components for each dataset (number of observations x dataset dimensions)
indep1 = np.random.randn(nSamples, 400)
indep2 = np.random.randn(nSamples, 500)

# Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
#data1 = 0.25*indep1 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2)).T
#data2 = 0.25*indep2 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T
data1 = 0.25*indep1 
data2 = 0.25*indep2 
# Split each dataset into two halves: training set and test set
train1 = data1[:int(nSamples/2)]
train2 = data2[:int(nSamples/2)]
test1 = data1[int(nSamples/2):]
test2 = data2[int(nSamples/2):]

##
def lc_rcca(datasets,kernelcca =True,reg=0.1,numCC=2,verbose=False):
#    datasets contain 2 subsets: X and Y
    cca = rcca.CCA(kernelcca =kernelcca, reg =reg, numCC =numCC )
    cca.train(datasets)
    # calc the correlation between the first cannonical variate
    corr_firstVariate=cca.__dict__['cancorrs'][0]
    return corr_firstVariate,cca

##    
def lc_rcca_CV_1fold(datasets_cv,kernelcca,regs,numCCs):
    # cross-validation
    # run
    # split datasets to train set and test set
#    datasets_cv=split_datasets(datasets,prop=2/3)
    corr=[]
    for  numCC in numCCs:
        corr_inner=[]
        for reg in regs:
            corr_firstVariate,_=lc_rcca(datasets_cv,kernelcca, reg, numCC,verbose=False)
            corr_inner.append(corr_firstVariate)
        corr.append(corr_inner)        
    return corr
##
def lc_rcca_CV_all_fold(datasets,K,kernelcca,regs,numCCs,n_processes=5):
    s=time.time()
#==========================================================
    Corr=[]
    if K>20:
        pool = multiprocessing.Pool(processes=n_processes)
        for k in range(K):
            # split datasets to train set and test set
            datasets_cv=split_datasets(datasets,prop=2/3)
            Corr.append(pool.apply_async(lc_rcca_CV_1fold,\
                             (datasets,kernelcca,regs,numCCs))) 
        print ('Waiting...')
        pool.close()   
        pool.join()
    CORR=[co.__dict__['_value'] for co in Corr]
    meanCorr=np.mean(CORR,0)
    e=time.time()
    print('parameter tuning time is {:.1f} s'.format(e-s))
    return meanCorr
#==========================================================
    if K<=20:
        Corr=[]
        for i in range(K):
            print('fold {}/{}'.format(i+1,K))
            # split datasets to train set and test set
            datasets_cv=split_datasets(datasets,prop=2/3)
            # run
            corr=lc_rcca_CV_1fold(datasets_cv,kernelcca=kernelcca,\
                       regs=regs,numCCs=numCCs)
            Corr.append(corr)
    meanCorr=np.mean(Corr,0)
    e=time.time()
    print('parameter tuning time is {:.1f} s(1)'.format(e-s))
    return meanCorr
#==========================================================

        
##        
def split_datasets(datasets,prop=2/3):
    # Only applicable to 2 datasets
    # prop: proportion of datasets used for cv
#    nData=len(datasets)
    nSample=datasets[0].shape[0]
    nCV=int(nSample*prop)
    index=np.random.permutation(np.arange(0,nSample,1))
    datasets_cv=[datasets[0][index[:nCV],:],datasets[1][index[:nCV],:]]
    return datasets_cv
    
if __name__=='__main__':
    datasets=[train1,train2]
    mc=lc_rcca_CV_all_fold(datasets,K=22,\
                        kernelcca=True,\
                        regs=np.logspace(-4,2,10),\
                        numCCs=np.arange(1,6),\
                        n_processes=5)