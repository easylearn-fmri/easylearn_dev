# -*- coding: utf-8 -*-
"""
Hosmer-Lemeshow test

@author: Alex (stackoverflow)
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2

def hosmer_lemeshow_test(pihat,real_label):
    # pihat=model.predict()
    pihatcat=pd.cut(pihat, np.percentile(pihat,[0,25,50,75,100]),labels=False,include_lowest=True) #here I've chosen only 4 groups

    meanprobs =[0]*4 
    expevents =[0]*4
    obsevents =[0]*4 
    meanprobs2=[0]*4 
    expevents2=[0]*4
    obsevents2=[0]*4 

    for i in range(4):
       meanprobs[i]=np.mean(pihat[pihatcat==i])
       expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
       obsevents[i]=np.sum(real_label[pihatcat==i])
       meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
       expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
       obsevents2[i]=np.sum(1-real_label[pihatcat==i]) 


    data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
    data2={'expevents':expevents,'expevents2':expevents2}
    data3={'obsevents':obsevents,'obsevents2':obsevents2}
    m=pd.DataFrame(data1)
    e=pd.DataFrame(data2)
    o=pd.DataFrame(data3)

    # The statistic for the test, which follows, under the null hypothesis,
    # The chi-squared distribution with degrees of freedom equal to amount of groups - 2. Thus 4 - 2 = 2
    tab = pd.DataFrame((np.array(o)-np.array(e))**2/np.array(e)).fillna(value=0).values
    tt=sum(sum(tab)) 
    pvalue=1-chi2.cdf(tt,2)

    return pd.DataFrame([[chi2.cdf(tt,2).round(2), pvalue.round(2)]],columns = ["Chi2", "p - value"])


if __name__ ==  "__main__":
    pred = np.array([0.6,0.7,0.1,0.6, 0.8,   0.2,0.1, 0.0,0.2,0.3])
    y = np.array([1,1,1,1,1,0,0,0,0,0,])
    dd=hosmer_lemeshow_test(pred, y)
    print(dd)