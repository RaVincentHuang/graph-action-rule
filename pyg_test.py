# Install required packages.
import os
import torch

os.environ["TORCH"] = torch.__version__
print(torch.__version__)

# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

import torch
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root="data/TUDataset", name="MUTAG")
# - root (str) – Root directory where the dataset should be saved.（保存的路径）
# - name (str) – The name of the dataset.（名字）

# 查看一些数据集的基本信息
print()
print(f"Dataset: {dataset}:")
print("====================")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

data = dataset[0]  # Get the first graph object.

print()
print(data)
print("=============================================================")

# 看一下第一张图的信息
# Gather some statistics about the first graph.
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")
print(f"Has self-loops: {data.has_self_loops()}")

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f"Step {step + 1}:")
    print("=======")
    print(f"Number of graphs in the current batch: {data.num_graphs}")
    print(data)
    print(f"Data.x.shape: {data.x.shape}, Data.x: {data.x}")
    print(f"Data.y.shape: {data.y.shape}, Data.y: {data.y}")
    print()

for step, data in enumerate(test_loader):
    print(f"Step {step + 1}:")
    print("=======")
    print(f"Number of graphs in the current batch: {data.num_graphs}")
    print(data)
    print()

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    # 把模型设置为训练模式

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        #  计算梯度
        optimizer.step()  # Update parameters based on gradients.
        #  根据上面计算的梯度更新参数
        optimizer.zero_grad()  # Clear gradients.
    #  清除梯度，为下一个批次的数据做准备，相当于从头开始


def test(loader):
    model.eval()
    # 把模型设置为评估模式

    correct = 0
    #  初始化correct为0，表示预测对的个数
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        #  预测的输出值
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        #  每个类别对应一个概率，概率最大的就是对应的预测值
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    #  如果一样，就是True，也就是1，correct就+1
    # 准确率就是正确的/总的
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
