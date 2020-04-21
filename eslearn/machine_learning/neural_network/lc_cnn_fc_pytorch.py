#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CNN for functional connectivity network

Created on Mon Jul 15 15:04:21 2019
This script is used to training validate and test a convolutional neural networ, finally using test data to test the model.
Validation data is used to optimize the super parameters. Test data must be used only once.
@author: Li Chao <lichao19870617@gmail.com>
If you think this code is useful, citing the easylearn software in your paper or code would be greatly appreciated!
Citing link: https://github.com/easylearn-fmri/easylearn  
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# Arguments
model_savepath = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data'
model_savename = 'model_7171806'
is_savefig = False
fig_savepath = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# CNN's Hyper parameters
num_epochs = 10
batch_size = 30
learning_rate = 0.001
in_channel = 1
num_classes = 2
cov1_kernel_size = [1, 114]
cov1_filter_number = 57
cov2_kernel_size = [114, 1]
cov2_filter_number = 120
fc1_node_number = 100
early_stopping = 0  # If the validation loss reached the first minimum, then stop training.



#%% utils
class Utils():
	"""
	Utilits for preprocessing data
	"""

	def __init__(self):
	    self.train_data_path = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\x_train304.npy'
	    self.label_train_path = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\y_train304.npy'

	    self.val_data_path = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\x_val76.npy'
	    self.val_label_path = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\y_val76.npy'

	    self.test_data_path = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\x_test206.npy'
	    self.test_label_path = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\y_test206.npy'

	def load_data(self):
	    """Load data and label
	    """

	    data_train = np.load(self.train_data_path)
	    label_train = np.load(self.label_train_path)[:, 1]

	    data_val = np.load(self.val_data_path)
	    label_val = np.load(self.val_label_path)[:, 1]

	    data_test = np.load(self.test_data_path)
	    label_test = np.load(self.test_label_path)[:, 0]

	    return data_train, label_train, data_val, label_val, data_test, label_test


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
	    n_node = 114
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


	def gen_gradient_label(self, label):
	    """
	    Encode label, and make label arange from zero to  N (N is the number of unique label)
	    """
	    uni_label = np.unique(label)
	    n_sample = np.int(label.shape[0])
	    
	    grd_label = np.zeros([n_sample,])
	    for i, ul in enumerate(uni_label):
	        grd_label[label == ul] = i
	    
	    return grd_label


	def prep_data(self, data_train, label_train, data_val, label_val, data_test, label_test):
	    """
	    Main function
	    1. normalization
	    2. 2d to 4d
	    3. transfer np.array to torch.tensor
	    """
	    label_train = self.gen_gradient_label(label_train)
	    label_val = self.gen_gradient_label(label_val)
	    label_test = self.gen_gradient_label(label_test)
	    # 1 normalization
	    data_train = self.normalization(data_train)
	    data_val = self.normalization(data_val)
	    data_test = self.normalization(data_test)

	    # 2 2d to 4d
	    data_train = self.trans_2d_to_4d(data_train)
	    data_val = self.trans_2d_to_4d(data_val)
	    data_test = self.trans_2d_to_4d(data_test)

	    # 3 transfer np.array to torch.tensor
	    data_train, label_train = torch.Tensor(data_train), torch.Tensor(label_train)
	    data_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False)
	    label_train = torch.utils.data.DataLoader(label_train, batch_size=batch_size, shuffle=False)

	    data_val, label_val = torch.Tensor(data_val), torch.Tensor(label_val)
	    data_test, label_test = torch.Tensor(data_test), torch.Tensor(label_test)
	    return data_train, label_train, data_val, label_val, data_test, label_test


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


	def plot_roc(self, real_label, predict_label,is_savefig=0, fig_savepath=os.getcwd, fig_savename = 'ROC.tif'):
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
	        plt.savefig(os.path.join(fig_savepath, fig_savename), dpi=300, bbox_inches='tight')
	    
	    return best_thresh


#%% Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
	"""
	This class is used to build CNN model
	"""
	def __init__(self, num_classes=2):
	    super(ConvNet, self).__init__()
	    self.layer1 = nn.Sequential(
	        nn.Conv2d(in_channel, cov1_filter_number,
	                  kernel_size=cov1_kernel_size, stride=1, padding=0),
	        nn.BatchNorm2d(cov1_filter_number),
	        nn.ReLU())
	        
	    self.layer2 = nn.Sequential(
	        nn.Conv2d(cov1_filter_number, cov2_filter_number,
	                  kernel_size=cov2_kernel_size, stride=1, padding=0),
	        nn.BatchNorm2d(cov2_filter_number),
	        nn.ReLU())
	        
	    self.fc1 = nn.Linear(cov2_filter_number, fc1_node_number)
	    self.dropout = nn.Dropout(p=0.4, inplace=False)
	    self.fc2 = nn.Linear(fc1_node_number, num_classes)
	    self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
	    out = self.layer1(x)
	    out = self.layer2(out)
	    out = out.reshape(out.size(0), -1)
	    out = self.fc1(out)
	    out = self.dropout(out)
	    out = self.fc2(out)
	    out = self.softmax(out)
	    return out


#%% training, validation and testing
class TrainTest(Utils):
	"""
	This class is used to train validation and testing.
	"""
	def __init__(self):
		super().__init__()

	def train_and_validation(self, data_train, label_train, data_val, label_val):
	    model = ConvNet(num_classes).to(device)

	    # Loss_train and optimizer
	    criterion = nn.CrossEntropyLoss()
	    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	    # Train the model
	    total_step = len(data_train)
	    Loss_train = []
	    Loss_validation = []
	    # Make sure save the model that trained at least one batch size sample.
	    # compare_loss is used to find the 
	    compare_loss = np.inf
	    # Marker of the minimum validation loss
	    marker = 0
	    
	    count = 0
	    for epoch in range(num_epochs):
	    	# marker==1 means already reached best model
	        if marker == 1:
	            break
	        
	        for i, (images, labels) in enumerate(zip(data_train, label_train)):
	            count += 1
	            images = images.to(device)
	            labels = labels.to(device)
	            
	            # Forward pass
	            model.train()
	            outputs = model(images)
	            
	            # tmp saving the model for saving the best model
	            model_beforetrain = model
	            
	            # make sure labels is scalar type long
	            labels = labels.long()
	            loss_train = criterion(outputs, labels)
	            
	            # Backward and optimize
	            optimizer.zero_grad()
	            loss_train.backward()
	            optimizer.step()

	            # Updata training loss
	            Loss_train.append(loss_train.item())

	            # Switch to evaluation mode and ***
	            # Updata validation loss
	            model.eval()
	            outputs_val = model(data_val)
	            label_val = label_val.long()
	            loss_val = criterion(outputs_val, label_val)
	            Loss_validation.append(loss_val.item())

	            if (i + 1) % 2 == 0:
	                # train loss_train
	                print('Epoch [{}/{}], Step [{}/{}], Loss_train: {:.4f}, loss_val: {:.4f}'
	                      .format(epoch + 1, num_epochs, i + 1, total_step, loss_train.item(), loss_val.item()))

	            # Save the model checkpoint
	            # if reach the first minimum validation loss
	            # then save the model
	            if early_stopping:
	                if (loss_val > compare_loss) and (marker != 1):
	                    marker = 1
	                    torch.save(model_beforetrain, os.path.join(model_savepath, ''.join([model_savename, '.ckpt'])))
	                    print(f'***Already found the best model***\nEarly Stoped!\n')
	                    break
	                else:
	                    compare_loss = loss_val

	    # if marker of the minimum validation loss is 0
	    # then save the last model          
	    if marker == 0 or (not early_stopping):
	        model_beforetrain = model
	        torch.save(model_beforetrain, os.path.join(model_savepath, ''.join(['model_', str(count), '.ckpt'])))


	    return model_beforetrain, Loss_train, Loss_validation


	def test_aftertrain(self, model, data, label):
	    """
	    Test the model just after trained the model
	    """
	    criterion = nn.CrossEntropyLoss()
	#    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
	    with torch.no_grad():
	        data = data.to(device)
	        # make sure labels is scalar type long
	        label = label.long()
	        label = label.to(device)

	        outputs = model(data)
	        loss_train = criterion(outputs, label)
	        _, predicted = torch.max(outputs.data, 1)

	        accuracy, sensitivity, specificity, auc_score = self.eval_prformance(
	            label, predicted, outputs)

	    return loss_train, accuracy, sensitivity, specificity, auc_score


	def validata(self, model, data, label):
	    """
	    Idengity the best thresh (max yuden idex) using pre-trained model(load from local device)
	    """
	    output = model(data)
	    _, predicted = torch.max(output.data, 1)
	    accuracy, sensitivity, specificity, auc_score = self.eval_prformance(label, predicted, output.data)
	    
	    best_thresh = self.plot_roc(label, output.data[:,1], is_savefig=is_savefig,fig_savepath=fig_savepath, fig_savename='ROC_val.tif')
	                
	    return accuracy, sensitivity, specificity, auc_score, best_thresh


	def test_using_best_thresh(self, model, thresh, data, label):
	    """
	    Test model using the best thresh
	    and plot ROC too
	    """
	    output = model(data)
	    predict = np.array(output[:,1].data) > thresh
	    predict = np.array([np.int(predict_) for predict_ in predict])
	    accuracy, sensitivity, specificity, auc = self.eval_prformance(label, predict, output.data)
	#    confusionmatrix = confusion_matrix(label, predict)
	    self.plot_roc(label, output.data[:,1], is_savefig=is_savefig, fig_savepath=fig_savepath, fig_savename='ROC_test.tif')

	    return accuracy, sensitivity, specificity, auc


#%%
if __name__ == '__main__':
    utils = Utils()
    traintest = TrainTest()
    
    # load
    data_train, label_train, data_val, label_val, data_test, label_test = utils.load_data()

    # prepare data
    data_train, label_train, data_val, label_val, data_test, label_test = \
        utils.prep_data(data_train, label_train, data_val,
               label_val, data_test, label_test) 
        
    # training
    model, Loss_train, Loss_validation = traintest.train_and_validation(
        data_train, label_train, data_val, label_val)

    # test model
#    model = torch.load(os.path.join(model_savepath, ''.join([model_savename, '.ckpt'])))
    accuracy_val, sensitivity_val, specificity_val, auc_score_val, thresh = traintest.validata(model, data_val, label_val)
    print(f'validation:acc/sens/spec/auc are\n {accuracy_val}, {sensitivity_val}, {specificity_val}, {auc_score_val}\n')
    loss_test, accuracy_test, sensitivity_test, specificity_test, auc = traintest.test_aftertrain (model, data_test, label_test)
    print(f'test: acc/sens/spec/auc are\n {accuracy_test}, {sensitivity_test}, {specificity_test},{auc}\n')
    
    #%% show
    # show loss_train
    plt.figure()
    plt.plot(Loss_train)
    plt.plot(Loss_validation)
    plt.legend(['Training','Validation'])
    plt.title('Loss')
    plt.show()
    
    # plot bar (val and test)
    plt.figure(figsize=(5,6))
    X = np.arange(3) + 1
    Y1 = [accuracy_val, sensitivity_val, specificity_val]
    Y2 = [accuracy_test, sensitivity_test, specificity_test]
    plt.bar(X,Y1,width = 0.35)
    
    #width
    plt.bar(X+0.35,Y2,width = 0.35)
    for x,y in zip(X,Y1):
        plt.text(x+0., y+0.05, '%.2f' % y, ha='center', va= 'bottom')
        
    for x,y in zip(X,Y2):
        plt.text(x+0.3, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
    plt.ylim(0,+1.25)
    
    ax = plt.gca()
    ax.legend([u'validation',u'test'])
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['right'].set_visible(False) #  
    plt.savefig(os.path.join(fig_savepath, 'bar.tif'), dpi=300, bbox_inches='tight')  
    plt.show()
