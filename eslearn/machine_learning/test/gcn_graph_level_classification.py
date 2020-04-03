"""
https://blog.csdn.net/skj1995/article/details/103780873
"""
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data

# Generate Data
def generate_data():
    edge_index = torch.tensor([[0, 1, 1, 2],[1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    y = torch.tensor([1], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

dataset = [generate_data() for i in range(100)]

# # Download data
# # path = osp.join(osp.dirname(osp.realpath('__file__')), 'data', 'ENZYMES')
# # dataset = TUDataset(path, name='ENZYMES')
# dataset = dataset.shuffle()
# #print("dataset.num_classes=",dataset.num_classes)

n = len(dataset) // 2
test_dataset = dataset[:n]  
train_dataset = dataset[n:]  
test_loader = DataLoader(test_dataset, batch_size=1) 
train_loader = DataLoader(train_dataset, batch_size=20)

# View data
# for data_for_check in train_loader:
#     print("data_for_check=",data_for_check)
#     print("data_for_check.batch=",data_for_check.batch)
#     print("data_for_check.batch.shape=",data_for_check.batch.shape)
#     print("data_for_check.x.shape=",data_for_check.x.shape)
#     print("data_for_check.num_features=",data_for_check.num_features)
#     print("\n")

# Net
class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
 
        self.conv1 = GraphConv(num_features, 128)  # num_features表示节点的特征数，为3
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
 
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)
 
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print("raw_x.shape=",x.shape)
        x = F.relu(self.conv1(x, edge_index))
        print("conv1_x.shape=",x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        print("x.shape=",x.shape)
        
        x_gmp=gmp(x, batch)
        x_gap=gap(x, batch)
        print("x_gmp.shape=",x_gmp.shape)
        print("x_gap.shape=",x_gap.shape)
        x1 = torch.cat([x_gmp, x_gap], dim=1)  # 表示在列上合并，增加更多的列
        print("x1.shape=",x1.shape)
 
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        print("x2.shape=",x2.shape)
 
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        print("x3.shape=",x3.shape)
 
        x = x1 + x2 + x3
        print("x+.shape=",x.shape)
 
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        print("final_x.shape",x.shape)
        return x
 
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(1, 20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
 
 
def train(epoch):
    model.train()
 
    loss_all = 0
    for data in train_loader:
        print("raw_data.y.shape",data.y.shape)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # data.y = data.y.reshape([])
        print("output.shape=", output.shape)
        print("data.y.shape=", data.y.shape)
        data.y = data.y.long()
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        print("\n")
    return loss_all / len(train_dataset)
 
 
def test(loader):
    model.eval()
 
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        print("\n")
    return correct / len(loader.dataset)
 
 
for epoch in range(1):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, test_acc))
    print("\n")
