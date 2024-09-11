import sys
sys.path.append('/home/vincent/graphrule/src')

from graph.model import GraphClassifier, train_loop, GraphClassifier2
from graph.store import SubgraphDataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


path = "/home/vincent/graphrule/data/subgraph/24point_2samplerrandom_walk-total_num6400-node_num10-node_featurenode.pt"
data: SubgraphDataset = torch.load(path)
# print(f"load data {data}")
# 查看 data.y 的分布
labels = [d.y.item() for d in data]
# print(f"labels: {labels}")
label_counts = torch.tensor(labels).bincount()
print(f"Label distribution: {label_counts}")

train_ratio = 0.8
train_size = int(len(data) * train_ratio)
test_size = len(data) - train_size
train_dataset, test_dataset = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = GraphClassifier(data[0].x.shape[1], 128, 4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(f"start train")
train_loop(model, train_loader, test_loader, optimizer, device, 100)
