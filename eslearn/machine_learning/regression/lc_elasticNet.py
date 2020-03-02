# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 16:31:20 2018
This is a class of enastic net regression
Refer to {"Individualized Prediction of Reading Comprehension
Ability Using Gray Matter Volume"}
{Elastic net with nesting cross-validationt alpha=lambda l1_ratio=lasso 惩罚系数}
Input:
    X: features
    y: responses (continuous variable)
    alphas: corresponds to the lambda (pow(np.e,np.linspace(-6,5,20)))
    l1_ratio: when= 1 elastic net is the lasso penalty (np.linspace(0.2,1,10))
    n_jobs: number of CPUs to perform CV
Output: 
    predict=predict responses/dependent variable
    y_sorted=sorted original responses
    Coef=predict coef
    r=Pearson's correlation coefficients between y_sorted and Predict
@author: Li Chao 
         Email:lichao19870617@gmail.com
"""


# search path append
import sys  
sys.path.append(r'D:\myCodes\LC_MVPA\Python\MVPA_Python\utils')
### import module
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
import sys,time
import numpy as np
import scipy.stats.stats as stats
# my module
from lc_splitX_accord_sorted_y  import splitX_accord_sorted_y
import lc_pca as pca
import lc_scaler as scl

# sample data
n_sample,n_features =1000,5000
np.random.seed(0)
coef = np.random.random(n_features)
#coef = np.array([[.8],[.1],[0.2]])
coef[100:] = 0.0  
X = np.random.random([n_sample, n_features])
np.random.randint(0,10)
y = np.dot(X, coef)+np.random.random([n_sample,])
#a=np.corrcoef(y[:,0],X[:,2])
#import lc_elasticNet as enet
#e=enet.elasticNet()
#Predict,y_sorted,Coef,r,p=e.elasticNetCV_Outer(X,y)
#scatter_LC(Predict,y_sorted)
#### class and def

# class
class elasticNet():
    # initial parameters
    def __init__(self,k=3,\
                 alphas=pow(np.e,np.linspace(-6,5,20)),\
                 l1_ratio=np.linspace(0.2,1,10),\
                 num_jobs=10,\
                 scale_method='StandardScaler',\
                 pca_n_component=0.9,\
                 permutation=0):
        self.k=k
        self.alphas=alphas
        self.l1_ratio=l1_ratio
        self.num_jobs=num_jobs
        self.scale_method=scale_method
        self.pca_n_component=pca_n_component
        self.permutation=permutation
#        print(self.alphas)
    # main function
    def elasticNetCV_Outer(self,X,y):
        #为了让sklearn处理，label不能是2D
        y=np.reshape(np.array(y),len(y))        
        # pre-allocating
        n_samples,n_features=X.shape
        Predict=np.array([])
        y_sorted=np.array([])
        Coef=np.empty([n_features,1])
        #
        if not self.permutation:start_time=time.clock()
        # obtian split index
        ind_orig=splitX_accord_sorted_y(y,self.k)
        ind_orig=np.array(ind_orig)
        for i in range(self.k):
            if not self.permutation:
                print('{}/{}'.format(i,self.k))
            # 1 split
            # X
            test_X=X[ind_orig[i],:]
            train_X=X
            train_X=np.delete(train_X,[ind_orig[i]],axis=0)
            # y
            test_y=y[ind_orig[i]]
            train_y=y
            train_y=np.delete(train_y,[ind_orig[i]],axis=0)
            ## 2 scale(optional)
            train_X,model=scl.scaler(train_X,self.scale_method)
            test_X=model.transform(test_X)
#            ## 3 reducing dimension(optional)
            if self.pca_n_component!=0:
#                s=time.time()
                train_X,trained_pca=pca.pca(train_X,self.pca_n_component)
#                e1=time.time()
                test_X=trained_pca.transform(test_X)
#                e2=time.time()
#                print('pca time is {} + {} and comp num is {}'.format(s-e1,s-e2,test_X.shape[1]))
            ## 4 parameter optimization
#            s=time.time()
            (optimized_alpha,optimized_l1_ratio,_,_)=\
            self.elasticNetCV_Inner(train_X,train_y,\
                                 self.alphas,\
                                 self.l1_ratio,\
                                 self.num_jobs)
#            e=time.time()
#            print('optimized time is {}'.format(s-e))
            ## 4 train
#            s=time.time()
            enet=self.elasticNet_OneFold(train_X,train_y,\
                                  optimized_alpha,optimized_l1_ratio)
#            e=time.time()
#            print('train time is {}'.format(s-e))
            # 5 test
            predict=enet.predict(test_X)  
            Predict=np.append(Predict,predict)
            coef=enet.coef_
            coef=coef.reshape(len(coef),1)
            if self.pca_n_component!=0:
                coef=coef.reshape(1,len(coef))
                coef=trained_pca.inverse_transform(coef)
                coef=coef.reshape(coef.size,1)
            Coef=np.hstack([Coef,coef])
            y_sorted=np.append(y_sorted,test_y)
            # 6 iteration/repeat
        if not self.permutation:
            end_time=time.clock()  
            print('running time is {:.1f} second'.format(end_time-start_time))
        r,p=stats.pearsonr(Predict,y_sorted)
        if not self.permutation:
            print('pearson\'s correlation coefficient r={:.3},p={}'.format(r,p))
        Coef=np.delete(Coef,0,axis=1)
        return Predict,y_sorted,Coef,r,p
           
    ###
    def elasticNetCV_Inner(self,X,y,alphas,l1_ratio,num_jobs):    
    
        enet = ElasticNetCV(cv=5,\
                        random_state=0,\
                        alphas=alphas,\
                        l1_ratio=l1_ratio,\
                        n_jobs=num_jobs)
        enet.fit(X, y)
    #    intercept=enet.intercept_
        optimized_alpha=enet.alpha_
        optimized_l1_ratio=enet.l1_ratio_
        optimized_coef=enet.coef_
        predict=enet.predict(X)
    #    mean_mse=enet.mse_path_    
        return optimized_alpha,optimized_l1_ratio,optimized_coef,predict
    
    ###
    def elasticNet_OneFold(self,X,y,alpha,l1_ratio):    
        enet = ElasticNet(random_state=0,alpha=alpha,l1_ratio=l1_ratio)
        enet.fit(X, y)    
        return enet