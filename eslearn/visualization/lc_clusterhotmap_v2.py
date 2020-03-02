import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import matplotlib.gridspec
import pandas as pd
import numpy as np


data = pd.read_excel(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Plot\data.xlsx')

kind = data.pop('KIND')
lut = dict(zip(kind.unique(), np.random.randn(20,3)))
row_colors = kind.map(lut)


data.index = data.iloc[:,0]
data = data.iloc[:,2:]
clname = list(data.columns)
data = data[['q_A',
             'q_A_unmedicated',
             'q_A_medicated',
             'q_B',
             'q_B_unmedicated',
             'q_B_medicated',
             'q_C',
             'q_C_unmedicated',
             'q_C_medicated',]]




#First create the clustermap figure
g = sns.clustermap(data, row_colors= np.random.randn(94,3), figsize=(13,8))
# set the gridspec to only cover half of the figure
#g.gs.update(left=0.05, right=0.45)
#
##create new gridspec for the right part
#gs2 = matplotlib.gridspec.GridSpec(1,1, left=0.6)
## create axes within this new gridspec
#ax2 = g.fig.add_subplot(gs2[0])
## plot boxplot in the new axes
#sns.boxplot(data=iris, orient="h", palette="Set2", ax = ax2)
plt.show()

np.random.randint(0,256,3)
