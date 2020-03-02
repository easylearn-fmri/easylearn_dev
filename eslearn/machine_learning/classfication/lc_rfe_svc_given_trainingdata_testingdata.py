# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:25:57 2019
@author: LI Chao
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
sys.path.append(r'F:\黎超\dynamicFC\Code\lc_rsfmri_tools_python-master\Machine_learning\classfication')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')
import numpy as np
from lc_read_nii import read_multiNii_LC
from lc_read_nii import read_sigleNii_LC
from lc_svc_rfe_cv_V2 import SVCRefCv

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
                 patients_path=r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\mALFF\patient_mALFF',  # 训练组病人
                 hc_path=r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\mALFF\control_mALFF',  # 训练组正常人
                 val_path=r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\mALFF\control_mALFF',  # 验证集数据
                 val_label=r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Machine_learning\classfication\val_label.txt',  # 验证数据的label文件
                 suffix='.img',  #图像文件的后缀
                 mask=r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\mALFF\patient_mALFF\mALFFMap_sub006.img',
                 k=2  # 训练集内部进行RFE时，用的kfold CV
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
        data1,_=read_multiNii_LC(self.patients_path, self.suffix)
        data1=np.squeeze(np.array([np.array(data1).reshape(1,-1) for data1 in data1]))
        data2,_=read_multiNii_LC(self.hc_path, self.suffix)
        data2=np.squeeze(np.array([np.array(data2).reshape(1,-1) for data2 in data2]))
        data=np.vstack([data1,data2])
        
        # validation data
        data_validation,self.name_val=read_multiNii_LC(self.val_path, self.suffix)
        data_validation=np.squeeze(np.array([np.array(data_validation).reshape(1,-1) for data_validation in data_validation]))
        
        # data in mask
        mask,_=read_sigleNii_LC(self.mask)
        mask=mask>=0.2
        mask=np.array(mask).reshape(-1,)
        
        self.data_train=data[:,mask]
        self.data_validation=data_validation[:,mask]
        
        # label_tr
        self.label_tr=np.hstack([np.ones([len(data1),])-1,np.ones([len(data2),])])
        print("loaded")
        return self

    def tr_te_ev(self):
        """
        训练，测试，评估
        """
        
        # scale
        data_train,data_validation=self.scaler(self.data_train,self.data_validation,self.scale_method)
        
        # reduce dim
        if 0<self.pca_n_component<1:
            data_train,data_validation,trained_pca=self.dimReduction(data_train,data_validation,self.pca_n_component)
        else:
            pass
        
        # training
        print("training...\nYou need to wait for a while")
        model,weight=self.training(data_train,self.label_tr,\
                 step=self.step, cv=self.k,n_jobs=self.num_jobs,\
                 permutation=self.permutation)

        # fetch orignal weight
        if 0 < self.pca_n_component< 1:
            weight=trained_pca.inverse_transform(weight)
        self.weight_all=weight
        
        # testing
        print("testing...")
        self.predict,self.decision=self.testing(model,data_validation)

        # eval performances
        self.val_label=np.loadtxt(self.val_label)
        self.eval_prformance(self.val_label,self.predict,self.decision)
        
        return self
    
    def main(self):
        self.load_data()
        self.tr_te_ev()
        return self
    
if __name__=="__main__":
    svc=SvcForGivenTrAndTe()
    results=svc.main()
    results=results.__dict__
    print("Done!\n")