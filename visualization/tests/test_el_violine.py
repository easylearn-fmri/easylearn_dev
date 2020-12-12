import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
from eslearn.visualization.el_violine import ViolinPlotMatplotlib, ViolinPlot

data = pd.read_excel("./violin_demo_data.xlsx")
var_name = list(data.columns)

data_list = list(data.values.T)

ViolinPlotMatplotlib().plot(data_list, 
                            positions=np.arange(0,len(data_list)))

# ViolinPlot().plot(data_list)


plt.xticks(np.arange(0,len(data_list)), var_name, rotation=45, color='w', fontsize=15)
plt.tight_layout()
mplcyberpunk.add_underglow()
plt.show()

plt.savefig("violin.pdf")