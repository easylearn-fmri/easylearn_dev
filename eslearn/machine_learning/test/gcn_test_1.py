"""
https://blog.csdn.net/skj1995/article/details/103780873
"""
import os.path as osp
import scipy.io as sio
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data

# read data
a=sio.loadmat(r'D:\My_Codes\easylearn-fmri\eslearn\machine_learning\test\GCNNCourseCodes\enzymes.mat')
# list of adjacency matrix
mat = a['A'][0]
# list of features
features = a['F'][0]
# label of graphs
target = a['Y'][0]
# test train index for 10-fold test
TRid=a['tr']
TSid=a['ts']


# Generate Data
def generate_data(x, m, y):
    m = np.where(m)
    m = np.array(m)
    edge_index = torch.tensor(m, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor([y], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

dataset = [generate_data(x, m, y) for (x, m, y) in zip(features, mat, target)]
perm = torch.randperm(len(dataset))

dataset1 = [dataset[i] for i in perm]
dataset = dataset1
del dataset1


n = len(dataset) // 10
test_dataset = dataset[:n]  
train_dataset = dataset[n:]  
test_loader = DataLoader(test_dataset, batch_size=60) 
train_loader = DataLoader(train_dataset, batch_size=20)

# # View data
# for data_for_check in train_loader:
#     print("data_for_check=",data_for_check)
#     print("data_for_check.batch=",data_for_check.batch)
#     print("data_for_check.batch.shape=",data_for_check.batch.shape)
#     print("data_for_check.x.shape=",data_for_check.x.shape)
#     print("data_for_check.num_features=",data_for_check.num_features)
#     print("\n")

# Net
# Construct network
class Net(torch.nn.Module):
    def __init__(self, num_feature, num_class):
        super(Net, self).__init__()
        self.conv1 = GraphConv(num_feature, 128)
        self.conv2 = GraphConv(128, 64)
        
        self.lin1 = torch.nn.Linear(64, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
 
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(3, 6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
 
 
def train(epoch):
    model.train()
 
    loss_all = 0
    for data in train_loader:
        # print("raw_data.y.shape",data.y.shape)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print("output.shape=", output.shape)
        # print("data.y.shape=", data.y.shape)
        data.y = data.y.long()
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)
 
 
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        _, pred = model(data).max(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
 
    
for epoch in range(10):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, train_acc, test_acc))
