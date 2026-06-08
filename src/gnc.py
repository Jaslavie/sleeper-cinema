"""
Graph Convolutional Network (GCN) based encoder
    - Input: Adjacency matrix A and feature matrix X
    - Output: Embedding matrix Z of each movie, updated weights
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device):
        super().__init__()

        self.device = device

        self.fc1 = GCNConv(in_features, hidden_features)
        self.fc2 = GCNConv(hidden_features, out_features)

    def forward(self, A, X):
        """
        Encode each film based on its neighbors and self
            X = N films x F features
            A = N films x N films adjacency matrix

        Returns:
            Z = N films x d embeddings
        """
        # process embeddings on device
        X = X.to(self.device)
        A = A.to(self.device)

        Z = self.fc1(X, A) # (N, embedding_dim)
        Z = F.relu(Z)

        Z = F.dropout(Z, p=0.5)

        Z = self.fc2(Z)

        return Z