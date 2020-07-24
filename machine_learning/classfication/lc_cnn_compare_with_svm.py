# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 16:56:08 2018

@author: lenovo
"""

from sklearn.svm import SVC
clf = SVC()
clf.fit(sel.x_train, sel.y_train) 
pred=clf.predict(test_data)

pred[pred==1]=0
pred[pred==3]=1
a=pred-test_label.T
a=a.T
sum(a==0)/206