import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.data
from torch_geometric.logging import log
from torch_geometric.nn import GraphConv, global_mean_pool, GATConv
import torch_geometric.visualization


class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GraphClassifier, self).__init__()

        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        x = global_mean_pool(x, batch=data.batch)
        x = self.linear(x)
        
        return x
    
def train(model, loader, optimizer):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def test(model, loader):
    model.eval()

    correct = 0
    #  初始化correct为0，表示预测对的个数
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())   # Derive ratio of correct predictions.

def train_loop(model, train_loader, test_loader, optimizer, criterion, epochs):
    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer)
        test_loss = test(model, test_loader)
        log(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}')
        
data = torch_geometric.data.Data()
data.x = torch.randn(100, 16)
data.edge_index = torch.randint(0, 100, (2, 100))



if __name__ == "__main__":
    model = GraphClassifier(1536, 64, 4)
    print(model)
