# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:04:21 2019
This script is used to training a convolutional neural network,
then validate and test this model using validation and test data
@author: Li Chao
Email: lichao19870617@gmail.com
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# parameters
n_node = 246  # number of nodes in the FC network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#%% utils
class Utils():
	"""
	Utilits for preprocessing data
	"""
	def __init__(self):
		pass

	def prep_data(self, test_data_path=None, n_node=None):
		"""
		Main function
		1. load data
		3. re_sampling
		4. normalization
		5. 2d to 4d
		6. transfer np.array to torch.tensor
		"""
		# load data and label
		self.test_data_path = test_data_path
		self.n_node = n_node
		feature_test, label_test = self.load_data()
		# 1 normalization
		feature_test = self.normalization(feature_test)
		# 2d to 4d
		feature_test = self.trans_2d_to_4d(feature_test)
		# 3 transfer np.array to torch.tensor
		feature_test, label_test = torch.Tensor(feature_test), torch.Tensor(label_test)
		return feature_test, label_test

	def load_data(self):
		"""
		Load data and label
		"""
		data_test = np.load(self.test_data_path)
		feature_test = data_test[:,1:]
		label_test = data_test[:,0]

		return feature_test, label_test

	def normalization(self, data):
		""" 
		Used to normalize (standardization) data (zero mean and one unit std: de-mean and devided by std)
		Because of our normalization level is on subject,
		we should transpose the data matrix on python(but not on matlab)
		"""
		scaler = preprocessing.StandardScaler().fit(data.T)
		z_data = scaler.transform(data.T) .T
		return z_data


	def trans_2d_to_4d(self, data_2d):
		""" 
		Transfer the 2d data to 4d data, so that the pytorch can handle the data
		"""
		n_node = self.n_node
		mask = np.ones([n_node, n_node])
		mask = np.triu(mask, 1) == 1  # 前期只提取了上三角（因为其他的为重复）
		data_4d = np.zeros([len(data_2d), 1, n_node, n_node])
		for i in range(data_4d.shape[0]):
			# 1) 将1维的上三角矩阵向量，嵌入到矩阵的上三角上
			data_4d[i, 0, mask] = data_2d[i, :]
			# 2) 将上三角矩阵，对称地镜像到下三角，从而组成完整的矩阵
			data_4d[i, 0, :, :] += data_4d[i, 0, :, :].T
			# 3) 对角线补上1
			data_4d[i, 0, :, :] += np.eye(n_node)
		return data_4d

	def eval_prformance(self, y_real, y_pred, y_pred_prob):
		"""
		Used to evaluate the model preformances
		"""
		accuracy = float('%.4f' % (accuracy_score(y_real, y_pred)))
		report = classification_report(y_real, y_pred)
		report = report.split('\n')
		specificity = report[2].strip().split(' ')
		sensitivity = report[3].strip().split(' ')
		specificity = float([spe for spe in specificity if spe != ''][2])
		sensitivity = float([sen for sen in sensitivity if sen != ''][2])
		auc_score = float('%.4f' % roc_auc_score(y_real, y_pred_prob[:, 1]))
		return accuracy, sensitivity, specificity, auc_score


	def plor_roc(self, real_label, predict_label,is_savefig=0, fig_savepath=os.getcwd, fig_savename = 'ROC.tif'):
		"""
		plot ROC and return the best threshold
		"""    
		fpr, tpr, thresh = roc_curve(real_label, predict_label)

		# identify best thresh (according yuden idex)
		yuden_idx = (1-fpr)+tpr-1
		loc = np.where(yuden_idx==np.max(yuden_idx))
		best_thresh = thresh[loc]

		plt.figure()
		plt.title('ROC Curve')
		plt.xlabel('False Positive Rate', fontsize=10)
		plt.ylabel('True Positive Rate', fontsize=10)
		plt.grid(True)
		plt.plot(fpr, tpr,'-')
		plt.plot(fpr[loc], tpr[loc],'o')

		# 设置坐标轴在axes正中心
		ax = plt.gca()
		ax.spines['top'].set_visible(False) #去掉上边框
		ax.spines['right'].set_visible(False) #
		ax.xaxis.set_ticks_position('bottom')  # 底部‘脊梁’设置为X轴
		ax.spines['bottom'].set_position(('data', 0))  # 底部'脊梁'移动位置，y的data
		ax.yaxis.set_ticks_position('left')  # 左部‘脊梁’设置为Y轴
		ax.spines['left'].set_position(('data', 0))  # 左部‘脊梁’移动位置，x的data
		# save figure
		if is_savefig:
			plt.savefig(os.path.join(fig_savepath, fig_savename), dpi=1200, bbox_inches='tight')

		return best_thresh

class TEST():
	"""
	This class is used to test the model using test data.
	"""
	def __init__(self):
		utils = Utils()

	def load_model(self, model_path=None):
		model = torch.load(model_path)
		return model

	def test(self, model, data, label):
	    """
	    Test the model just after trained the model
	    """
	#    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
	    with torch.no_grad():
	        data = data.to(device)
	        # make sure labels is scalar type long
	        label = label.long()
	        label = label.to(device)

	        outputs = model(data)
	        _, predicted = torch.max(outputs.data, 1)
	        accuracy, sensitivity, specificity, auc_score = utils.eval_prformance(label, predicted, outputs)

	    return outputs, predicted, accuracy, sensitivity, specificity, auc_score

#%%
if __name__ == '__main__':
    # prepare data
    utils = Utils()
    feature_test, label_test = utils.prep_data(test_data_path=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\val_54.npy',
    										   n_node=246) 
    # test model
    from lc_cnn_training_pytorch_v1 import ConvNet
    model = ConvNet()
    test = TEST()
    model = test.load_model(model_path = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\MLmodel\model_2019_11_26_20_40_51.ckpt')
    model.eval()
    outputs, predicted, accuracy_test, sensitivity_test, specificity_test, auc = test.test(model, feature_test, label_test)
    print(f'test: acc/sens/spec/auc are\n {accuracy_test}, {sensitivity_test}, {specificity_test},{auc}\n')