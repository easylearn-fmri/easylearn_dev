# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:25:57 2019
@author: LI Chao
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('Qt5Agg')

    
permutation_train_file = r'D:\workstation_b\xiaowei\TOLC_20200811\permutation_test_results_train.npy'
permutation_test_file = r'D:\workstation_b\xiaowei\TOLC_20200811\permutation_test_results_validation.npy'
permutation_train = np.load(permutation_train_file)
permutation_test = np.load(permutation_test_file)
accuracy_train = 0.84
accuracy_test = 0.67

perm_acc_tr = np.squeeze(permutation_train[0,:])
perm_acc_te = np.squeeze(permutation_test[0,:])
p_tr = (np.sum(perm_acc_tr >accuracy_train)+ 1)/ (len(perm_acc_tr)+1)
p_te = (np.sum(perm_acc_te >accuracy_test)+ 1)/ (len(perm_acc_te)+1)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(perm_acc_tr, bins=20, color='darkturquoise', alpha=0.6)
plt.plot([accuracy_train, accuracy_train],[0, 20], '--', linewidth=3, color='orange')
plt.xlabel('Random accuracy')
plt.ylabel('Frequence')
plt.title(f"Training set (p = {p_tr:.2f})")

plt.subplot(1,2,2)
plt.hist(perm_acc_te, bins=15, color='darkturquoise', alpha=0.6)
plt.plot([accuracy_test, accuracy_test],[0, 15], '--', linewidth=3, color='orange')
plt.xlabel('Random accuracy')
plt.ylabel('Frequence')
plt.title(f"Test set (p = {p_te:.3f})")


# Save to PDF format
plt.tight_layout()
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)  # wspace 左右
pdf = PdfPages(r'D:\workstation_b\xiaowei\TOLC_20200811\permutation_result.pdf')
pdf.savefig()
pdf.close()
plt.show()
