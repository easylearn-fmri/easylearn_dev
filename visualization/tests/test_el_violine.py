import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
from eslearn.visualization.el_violine import ViolinPlotMatplotlib, ViolinPlot

data = pd.read_excel(r"D:\software\conda\miniconda\envs\eslearn_env\Lib\site-packages\eslearn\visualization\tests/violin_demo_data-1.xlsx")
var_name = list(data)
data = np.array(data)
data[:,1] = data[:,2] - 0.4
data[:,2] = data[:,2] - 1

data_list = list(data.T)

ViolinPlotMatplotlib().plot(data_list, 
                            positions=np.arange(0,len(data_list)),
                            facecolor=["r","g","b"])

# ViolinPlot().plot(data_list)

plt.xticks(np.arange(0,len(data_list)), var_name, rotation=45, color='w', fontsize=15)
plt.ylabel("Ki-67",color='w', fontsize=15)
plt.tight_layout()
mplcyberpunk.add_underglow()
plt.show()

plt.savefig(r"D:\software\conda\miniconda\envs\eslearn_env\Lib\site-packages\eslearn\visualization\tests/violin.pdf")