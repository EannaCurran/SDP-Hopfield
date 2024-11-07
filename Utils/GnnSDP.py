# Basically the same as the baseline except we pass edge features
import torch
from torch_geometric import nn
import torch_geometric.nn
import torch.nn.functional as F


class GNNModel(torch.nn.Module):
    def __init__(self, num_features=1, hidden_size=32, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.convs = [torch_geometric.nn.GATConv(self.num_features, self.hidden_size,edge_dim=2, heads=1,dropout=0.1)]
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr = edge_attr.float()
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.tanh(x)

        x = self.linear(x)
        x = F.tanh(x)

        return x