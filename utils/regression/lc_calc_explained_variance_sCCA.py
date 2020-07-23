# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 20:31:13 2018
revised the rcca
this code is used to evaluate the sCCA model
@author: lenovo
"""
# search path append
import sys  
sys.path.append(r'D:\Github_Related\Github_Code\sCCA_Python\pyrcca-master')
from rcca import predict
import numpy as np

def lc_compute_ev(vdata,ws,verbose=1,cutoff=1e-15):
#    vdata is the validation datesets. ws is the weights
#    derived from train datesets
#    So, this function is used to validate the sCCA model
    nD = len(vdata)
#    nT = vdata[0].shape[0]
    nC = ws[0].shape[1]
    nF = [d.shape[1] for d in vdata]
    ev = [np.zeros((nC, f)) for f in nF]
    for cc in range(nC):
        ccs = cc+1
        if verbose:
            print('Computing explained variance for component #%d' % ccs)
        preds, corrs= predict(vdata, [w[:, ccs-1:ccs] for w in ws],
                               cutoff)
        resids = [abs(d[0]-d[1]) for d in zip(vdata, preds)]
        for s in range(nD):
            ev_ = abs(vdata[s].var(0) - resids[s].var(0))/vdata[s].var(0)
            ev_[np.isnan(ev_)] = 0.
            ev[s][cc,:] = ev_
    return ev