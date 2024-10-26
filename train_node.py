import sys
sys.path.append('/home/vincent/graphrule/src')

from graph.model import NodeClassifier
from graph.store import SubgraphDataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


path = "/home/vincent/graphrule/data/subgraph/nodes_samplerrandom_walk-total_num16000-node_num8-node_featurenode.pt"
data: SubgraphDataset = torch.load(path)

# scaler = StandardScaler()
# for graph in data:
#     graph['node'].x = torch.tensor(scaler.fit_transform(graph['node'].x), dtype=torch.float)
    
train_ratio = 0.8
train_size = int(len(data) * train_ratio)
test_size = len(data) - train_size
train_dataset, test_dataset = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = NodeClassifier(4, 64, 2, 4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"start train")
NodeClassifier.train_loop(model, train_loader, test_loader, optimizer, device, 1000)
torch.save(model.state_dict(), "/home/vincent/graphrule/model/node_classifier.pt")
