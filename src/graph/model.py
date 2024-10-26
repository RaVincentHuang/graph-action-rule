from httpx import head
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.data
from torch_geometric.logging import log
from torch_geometric.nn import GraphConv, global_mean_pool, GATConv, TopKPooling, global_max_pool, HeteroConv, BatchNorm, GATv2Conv
import torch_geometric.visualization

class GraphClassifier2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GraphClassifier2, self).__init__()
        
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.pool1 =  TopKPooling(hidden_channels, ratio=0.8)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)
        self.linear1 = torch.nn.Linear(hidden_channels * 2, hidden_channels // 4)
        self.linear2 = torch.nn.Linear(hidden_channels // 4, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        
        x = self.conv2(x, edge_index)
        x = x.relu()
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # x = self.conv3(x, edge_index)
        # x = self.conv3(x, edge_index)
        # x = x.relu()
        x = x1 + x2
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x).relu()
        
        return x
class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GraphClassifier, self).__init__()
        
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        # self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        # x = self.conv3(x, edge_index)
        # x = x.relu()
        
        x = global_mean_pool(x, batch=data.batch)
        x = self.linear(x)
        
        return x
    
    @staticmethod
    def test(model, loader, device):
        model.eval()

        correct = 0
        #  初始化correct为0，表示预测对的个数
        cnt = 0
        for data in loader:
            data = data.to(device) # Iterate in batches over the training/test dataset.
            out = model(data)
            pred = out.argmax(dim=1) # Use the class with highest probability.
            correct += int((pred == data.y).sum())
            cnt += pred.shape[0]
        
        return correct / cnt # Derive ratio of correct predictions.
    
    @staticmethod
    def train_loop(model, train_loader, test_loader, optimizer, device, epochs):
        for epoch in range(1, epochs + 1):
            loss = train(model, train_loader, optimizer, device)
            test_loss = GraphClassifier.test(model, test_loader, device)
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Test Accuracy: {test_loss:.4f}')
    

class NodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=2):
        super(NodeClassifier, self).__init__()
        
        self.conv1 = HeteroConv({
            ('node', 'add', 'node'): GATv2Conv(in_channels, hidden_channels, heads=heads),
            ('node', 'sub', 'node'): GATv2Conv(in_channels, hidden_channels, heads=heads),
            ('node', 'mul', 'node'): GATv2Conv(in_channels, hidden_channels, heads=heads),
            ('node', 'div', 'node'): GATv2Conv(in_channels, hidden_channels, heads=heads),
        })
        
        self.conv2 = HeteroConv({
            ('node', 'add', 'node'): GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads),
            ('node', 'sub', 'node'): GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads),
            ('node', 'mul', 'node'): GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads),
            ('node', 'div', 'node'): GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads),
        })
        
        self.norms1  = torch.nn.ModuleDict({node_type: BatchNorm(hidden_channels * heads) for node_type in ['node']})
        self.norms2  = torch.nn.ModuleDict({node_type: BatchNorm(hidden_channels * heads) for node_type in ['node']})

        self.linear1 = torch.nn.Linear(hidden_channels * heads, hidden_channels // 8)
        self.linear2 = torch.nn.Linear(hidden_channels // 8, num_classes)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        # x_dict = {key: self.norms1[key](x) for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        # x_dict = {key: self.norms2[key](x) for key, x in x_dict.items()}
        
        x = torch.cat([x_dict for x_dict in x_dict.values()], dim=1)
        x = self.linear1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        return x
    
    @staticmethod
    def test(model, loader, device):
        model.eval()

        correct = 0
        cnt = 0
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            cnt += pred.shape[0]

        return correct / cnt
    
    @staticmethod
    def train_loop(model, train_loader, test_loader, optimizer, device, epochs):
        for epoch in range(1, epochs + 1):
            loss = train(model, train_loader, optimizer, device)
            test_loss = NodeClassifier.test(model, test_loader, device)
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Test Accuracy: {test_loss:.4f}')
    
def train(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)



if __name__ == "__main__":
    model = NodeClassifier(4, 64, 2)
    print(model)
