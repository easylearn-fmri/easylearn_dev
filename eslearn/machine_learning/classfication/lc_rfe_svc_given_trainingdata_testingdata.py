# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:25:57 2019
@author: LI Chao
"""
import sys
import numpy as np
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import Memory
from shutil import rmtree

from eslearn.model_evaluation.el_evaluation_model_performances import eval_performance
from eslearn.utils.lc_niiProcessor import NiiProcessor
from lc_svc_rfe_cv_V2 import SVCRefCv
from eslearn.utils.lc_evaluation_model_performances import eval_performance

class SvcForGivenTrAndTe(SVCRefCv):
    """
    Training model on given training data.
    Then apply this mode to another testing data.
    Last, evaluate the performance
    If you encounter any problem, please contact lichao19870617@gmail.com
    """
    def __init__(self,
                 # =====================================================================
                 # all inputs are follows
                 patients_path=r'D:\workstation_b\xiaowei\ToLC\training\BD_label1',  # 训练组病人
                 hc_path=r'D:\workstation_b\xiaowei\ToLC\training\MDD__label0',  # 训练组正常人
                 val_path=r'D:\workstation_b\xiaowei\ToLC\testing',  # 验证集数据
                 val_label=r'D:\workstation_b\xiaowei\ToLC\testing_label.txt',  # 验证数据的label文件
                 suffix='.nii',
                 mask=r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii',
                 k=5  # 训练集内部进行RFE时，用的kfold CV
                 # =====================================================================
                 ):
        
        super().__init__()
        self.patients_path=patients_path
        self.hc_path=hc_path
        self.val_path=val_path
        self.val_label=val_label
        self.suffix=suffix
        self.mask=mask
        self.k=k
        print("SvcForGivenTrAndTe initiated")
        
    def _load_data_infolder(self):
        """load training data and validation data and generate label for training data"""
        print("loading...")
        # train data
        data1, _ = NiiProcessor().read_multi_nii(self.patients_path, self.suffix)
        data1 = np.squeeze(np.array([np.array(data1).reshape(1,-1) for data1 in data1]))
        data2,_ = NiiProcessor().read_multi_nii(self.hc_path, self.suffix)
        data2 = np.squeeze(np.array([np.array(data2).reshape(1,-1) for data2 in data2]))
        data = np.vstack([data1,data2])
        
        # validation data
        data_validation,self.name_val=NiiProcessor().read_multi_nii(self.val_path, self.suffix)
        data_validation=np.squeeze(np.array([np.array(data_validation).reshape(1,-1) for data_validation in data_validation]))
        
        # data in mask
        mask, _ = NiiProcessor().read_sigle_nii(self.mask)
        self.mask_orig = mask>=0.2
        self.mask_1d = np.array(self.mask_orig).reshape(-1,)
        
        self.data_train = data[:,self.mask_1d]
        self.data_validation = data_validation[:,self.mask_1d]
        
        # label_tr
        self.label_tr=np.hstack([np.ones([len(data1),]),np.ones([len(data2),]) - 1])
        print("loaded")
        return self

    def pipeline_grid(self, x_train, y_train):
        # Make pipeline
        location = 'cachedir'
        memory = Memory(location=location, verbose=10)
        pipe = Pipeline([
                ('reduce_dim', PCA()),
                ('feature_selection', SelectKBest(f_classif)),
                ('classify', LogisticRegression(solver='saga', penalty='l1'))
            ], 
            memory=memory
        )


        # In[6]:

        # Set paramters according to users inputs
        # PCA参数
        max_components = 0.99
        min_components = 0.3
        number_pc = 10
        range_dimreduction = np.linspace(min_components, max_components, number_pc).reshape(number_pc,)
        
        # ANOVA参数
        pca = PCA(n_components=min_components)
        pca.fit(X=x_train)
        min_number_anova = pca.n_components_
        pca = PCA(n_components=max_components)
        pca.fit(X=x_train)
        max_number_anova = pca.n_components_
        number_anova = 3
        range_feature_selection = np.arange(min_number_anova, max_number_anova, 10)

        # 分类器参数
        max_l1_ratio = 1
        min_l1_ratio = 0.5
        number_l1_ratio = 2
        range_l1_ratio = np.linspace(min_l1_ratio, max_l1_ratio, number_l1_ratio).reshape(number_l1_ratio,)

        # 整体grid search设置
        param_grid = [
            {
                'reduce_dim__n_components': range_dimreduction,
                'feature_selection__k': range_feature_selection,
                'classify__l1_ratio': [max_l1_ratio],
            },
        ]

        # In[ ]:

        # Train
        grid = GridSearchCV(pipe, n_jobs=-1, param_grid=param_grid)
        grid.fit(x_train, y_train)

        # In[8]:
        # Delete the temporary cache before exiting
        memory.clear(warn=False)
        rmtree(location)

        # In[9]:
        return grid

    def tr_te_ev(self):
        """训练，测试，评估
        """
        
        # scale
        data_train,data_validation=self.scaler(self.data_train,self.data_validation,self.scale_method)

        #%% training
        print("training...\nYou need to wait for a while")
        grid = self.pipeline_grid(data_train, self.label_tr)
        
        pred_train = grid.predict(data_train)
        dec_train = grid.predict_proba(data_train)[:,1]

        # fetch orignal weight
        estimator = grid.best_estimator_
        clf = estimator['classify']
        pca_model = estimator['reduce_dim']
        select_model = estimator['feature_selection']
        
        weight = np.zeros(np.size(select_model.get_support()))
        weight[select_model.get_support()] = clf.coef_[0]
        weight = pca_model.inverse_transform(weight)
        weight_3d = np.zeros(self.mask_orig.shape)
        self.weight_3d[self.mask_orig] = weight
        
        # testing
        print("testing...")
        self.predict = grid.predict(data_validation)
        self.decision = grid.predict_proba(data_validation)[1]

        # eval performances
        acc, sens, spec, auc = eval_performance(
            self.label_tr,pred_train,dec_train, 
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
            verbose=1, is_showfig=False,
        )

        self.val_label=np.loadtxt(self.val_label)
        acc, sens, spec, auc = eval_performance(
            self.val_label,self.predict,self.decision, 
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
            verbose=1, is_showfig=False,
        )

    
    def main(self):
        self._load_data_infolder()
        self.tr_te_ev()
    
if __name__=="__main__":
    svc=SvcForGivenTrAndTe()
    svc.main()
    print("Done!\n")