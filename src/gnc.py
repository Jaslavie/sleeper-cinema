"""
Graph Convolutional Network (GCN) based encoder. Specifically, this combines independent features of 
each movie and its relationship with neighbors. The resulting embeddings encode how a movie is semantically
similar and dissimilar to its neighbors.
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

        # Each convolution layer automatically updates weights during training
        self.fc1 = GCNConv(in_features, hidden_features)
        self.fc2 = GCNConv(hidden_features, out_features)

    def forward(self, A, X):
        """
        Encode each film based on its neighbors and self
            X = N films x F features
            A = N films x N films adjacency matrix

        Returns:
            H = N films x d embeddings
        """
        # process embeddings on device
        X = X.to(self.device) # (N, features_dim)
        A = A.to(self.device) # Graph

        H = self.fc1(X, A) # (N, embedding_dim)
        H = F.relu(H)
        H = F.dropout(H, p=0.5, training=self.training) # dropout during training only
        H = self.fc2(H, A)

        return H # (N, embedding_dim)