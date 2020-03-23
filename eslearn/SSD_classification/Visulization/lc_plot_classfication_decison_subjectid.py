# -*- coding: utf-8 -*-
"""
This script is used to plot classification 2D scatter.
X axis is decision_value, Y axis is subject number. from 1 to N.
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
import pickle

#%% Inputs
scale_550_file = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'
scale_206_file = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\SZ_NC_108_100-WF.csv'
scale_206_drug_file = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\SZ_109_drug.xlsx'
classification_results_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_pooling.npy'

# Load scales and results
scale_550 = pd.read_excel(scale_550_file)
scale_206 = pd.read_csv(scale_206_file)
scale_206_drug = pd.read_excel(scale_206_drug_file)
with open(classification_results_file, 'rb') as f1:
    results = pickle.load(f1)
results = pd.DataFrame(results['special_result'])

# Filter subjects that have .mat files
scale_550_selected = pd.merge(results, scale_550, left_on=0, right_on='folder', how='inner')
data_unmedicated_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['用药'] == 0)]
data_medicated_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['用药'] != 0)]

# Extract
decision = results.iloc[:, [0,2]]
loc_med =  np.isin(decision[0], data_unmedicated_550[0]) == False
decision_med = decision.iloc[loc_med][2]
decision_unmed = decision.loc[data_unmedicated_550.index][2]
decision = decision[2]
label_real =  results.iloc[:,1]
label_real_med = label_real.loc[data_medicated_550.index]
label_real_unmed = label_real.loc[data_unmedicated_550.index]

#%% Plot
# fig, ax = plt.subplots(1,2, figsize=(15,5))
plt.hist(decision_0, bins=20, alpha=1, color='lightblue')
plt.hist(decision_1, bins=20, alpha=1,color='paleturquoise')
plt.hist(decision_unmed, bins=20, alpha=1, color='darkturquoise')
plt.legend(['HC', 'SZ', 'First episode unmedicated SZ'])

# Save figure to PDF file
plt.tight_layout()
plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\decision_hist.pdf')
pdf.savefig()
pdf.close()
plt.show()
