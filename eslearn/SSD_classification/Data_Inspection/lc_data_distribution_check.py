"""
This script is used to check the data distribution across multiple datasets
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['savefig.dpi'] = 1200

# Input data
data_550_path = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_550.npy'
data_206_path = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_206.npy'
data_UCAL_path = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_UCLA.npy'
data_COBRE_path = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_COBRE.npy'

# Load data
data_550 = np.load(data_550_path)
data_206 = np.load(data_206_path)
data_UCAL = np.load(data_UCAL_path)
data_COBRE = np.load(data_COBRE_path)

data_550 = data_550[:, 2:]
data_206 = data_206[:, 2:]
data_UCAL = data_UCAL[:, 2:]
data_COBRE = data_COBRE[:, 2:]

# Calc mean
mean_data_550 = np.mean(data_550,axis=0)
mean_data_206 = np.mean(data_206,axis=0)
mean_data_UCAL = np.mean(data_UCAL,axis=0)
mean_data_COBRE = np.mean(data_COBRE,axis=0)

mean_all = np.vstack([mean_data_550, mean_data_206, mean_data_UCAL, mean_data_COBRE])
corrcoef = np.corrcoef(mean_all)


# Full Matrix
mask = np.triu(np.ones([246,246]),1) == 1
matrix_550 = np.zeros([246,246])
matrix_162 = np.zeros([246,246])
matrix_206 = np.zeros([246,246])
matrix_UCAL = np.zeros([246,246])
matrix_COBRE = np.zeros([246,246])

matrix_550[mask] = mean_data_550
matrix_206[mask] = mean_data_206
matrix_UCAL[mask] = mean_data_UCAL
matrix_COBRE[mask] = mean_data_COBRE

matrix_550 = matrix_550 + matrix_550.T
matrix_206 = matrix_206 + matrix_206.T
matrix_UCAL = matrix_UCAL + matrix_UCAL.T
matrix_COBRE = matrix_COBRE + matrix_COBRE.T

sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\matrix_550.mat', {'matrix_550': matrix_550})
sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\matrix_206.mat', {'matrix_206': matrix_206})
sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\matrix_UCAL.mat', {'matrix_UCAL': matrix_UCAL})
sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\matrix_COBRE.mat', {'matrix_COBRE': matrix_COBRE})
#%% -------------------------------Visualization------------------------------
# Show matrix
plt.figure(figsize=(20,7))

plt.subplot(2, 5, 1)
plt.imshow(matrix_550, cmap='jet',vmax=1, vmin=-0.5)
plt.grid(False)
plt.title('Dataset 1')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 5, 2)
plt.imshow(matrix_206, cmap='jet',vmax=1, vmin=-0.5)
plt.grid(False)
plt.title('Dataset 2')
plt.xticks([])
plt.yticks([])
plt.subplot(2, 5, 3)
plt.imshow(matrix_COBRE, cmap='jet',vmax=1, vmin=-0.5)
plt.grid(False)
plt.title('Dataset 3')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 5, 4)
plt.imshow(matrix_UCAL, cmap='jet',vmax=1, vmin=-0.5)
plt.grid(False)
plt.title('Dateset 4')
plt.xticks([])
plt.yticks([])

# Plot correlation heatmap
plt.subplot(2, 5, 6)
heatmap = sns.heatmap(corrcoef , annot=True, fmt = '.2f', cmap='jet')
plt.xticks()
plt.xticks([0.5, 1.5, 2.5, 3.5], ['Dataset 1',
                          'Dataset 2',
                          'Dataset 3', 'Dataset 4'], rotation = 45)
    
plt.yticks([0.5, 1.5, 2.5, 3.5], ['Dataset 1',
                          'Dataset 2',
                          'Dataset 3', 'Dataset 4'], rotation = 0)
plt.title('Correlations')
plt.show()

# Plot hist
plt.subplot(2, 5, 7)
sns.distplot(mean_data_550, bins=20, kde=True, color='darkturquoise')
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(2, 5, 8)
sns.distplot(mean_data_206, bins=20, kde=True, color='darkturquoise')
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(2, 5, 9)
sns.distplot(mean_data_COBRE, bins=20, kde=True, color='darkturquoise')
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(2, 5, 10)
sns.distplot(mean_data_UCAL, bins=20, kde=True, color='darkturquoise')
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save to PDF format
plt.tight_layout()
# pdf = PdfPages(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Figure\distribution1.pdf')
# pdf.savefig()
# pdf.close()
plt.show()