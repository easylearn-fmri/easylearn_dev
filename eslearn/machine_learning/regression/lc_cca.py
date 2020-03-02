# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:17:50 2018
典型相关分析
please refer to {Linked dimensions of psychopathology and 
connectivity in functional brain networks}
@author: Li Chao
"""

from sklearn.cross_decomposition import CCA
import numpy as np

class LcCCA():
    # initial parameters
    def __init__(self,n_comp=4,permutation=0):
        self.n_comp=n_comp
        self.permutation=permutation
        
   ##
    def cca(self,X,y):
        cca_model = CCA(n_components=self.n_comp,scale=False)
        cca_model.fit(X, y)
        X_c, y_c = cca_model.transform(X, y) 
        y_predict=cca_model.predict(X,copy=True) 
    #    R2=cca_model.score(X, y, sample_weight=None)
        
        # loading 为每个原始变量与对应典型变量的相关性*
        loading_x=cca_model.x_loadings_
        loading_y=cca_model.y_loadings_
    
        # weight即为线性组合的系数，可能可以用来将降维的变量投射到原始空间
        # 注意：如果scale参数设置为Ture，则weight是原始数据经过标准化后得到的weight
        weight_x=cca_model.x_weights_
        weight_y=cca_model.y_weights_
    #    weight_orig=np.dot(y_c[0,:],weight_y.T)
        
        # coef为X对y的系数，可以用来预测y（np.dot,矩阵乘法）
        coef=cca_model.coef_
        
        # 此算法中rotations==weight
#        rotation_y=cca_model.y_rotations_ 
#        rotation_x=cca_model.x_rotations_
        # score(X,y)返回R squre
        
        # 求某个典型变量对本组变量的协方差解释度（covariance explained by each canonical variate or component）
        cov_x= np.cov(X_c.T)
        cov_y= np.cov(y_c.T)
#        np.diag(cov_x)
        eigvals_x,_ = np.linalg.eig(cov_x)
        eigvals_y,_ = np.linalg.eig(cov_y)
        explain_x=pow(eigvals_x,2)/np.sum(pow(eigvals_x,2))
        explain_y=pow(eigvals_y,2)/np.sum(pow(eigvals_y,2))
    #    np.sort(explain)
        
        return (cca_model,\
                X_c,y_c,\
                loading_x,loading_y,\
                weight_x,weight_y,\
                explain_x,explain_y,\
                coef,y_predict)
    
    #==================================
if __name__=="__main__":
#    from sklearn.datasets import make_multilabel_classification
    import lc_cca as lcca
    n = 1000
    # 2 latents vars:
    l1 = np.random.normal(size=n)
    l2 = np.random.normal(size=n)
    
    latents = np.array([l1, l1, l2, l2]).T
    X = latents + np.random.normal(size=4 * n).reshape((n, 4))
    y = latents + np.random.normal(size=4 * n).reshape((n, 4))
#    y=np.random.permutation(y)
#    n_sample,n_features_x,n_features_y =500,4,2
#    np.random.seed(0)
#    coef = np.random.randn(n_features_x,n_features_y)
#    #coef = np.array([[.8],[.1],[0.2]]) 
#    X = np.random.random([n_sample, n_features_x])
#    y = np.dot(X, coef)
    #
    myCCA=lcca.LcCCA()
    (cca_model,X_c,y_c,\
            loading_x,loading_y,\
            weight_x,weight_y,\
            explain_x,explain_y,\
            coef,y_predict)=myCCA.cca(X,y)
#    np.corrcoef(y_predict[:,0],y[:,0])
    r=[]
    for i in range(X_c.shape[1]):
        r.append(np.corrcoef(X_c[:,i],y_c[:,i])[0,1])
    #
    bb=np.dot(y,weight_y)
    aa=np.dot(X,weight_x)