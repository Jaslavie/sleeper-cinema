"""
Generates and adds outlier movies from film embeddings (feature representation space) 
of normal movies onto graph.

These are used as negative (synthetic) examples during training and is not used during inference.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

from graph import Graph
class OutlierGenerator(nn.Module):
    """
    Generates pseudo-anomaly embeddings from labeled-normal film embeddings.
    Note that real anomalies (labeled as 1) are completely held out during this process.

    Inputs expected at training time:
        H_normal: encoded film embeddings from the GCN, shape [N, out_features]
        A (graph): graph adjacency matrix encodes neighbors for each node
        normal_mask: boolean/index mask for normal films

    Output:
        H_outlier: generated outlier embeddings, shape [S, out_features], S = number of outliers
    """

    def __init__(
        self, 
        out_features, 
        device, 
        outlier_ratio, 
        # gaussian_mean, 
        # gaussian_std
    ):
        super().__init__()
        self.device = device
        self.outlier_ratio = outlier_ratio
        self.transform = nn.Linear(out_features, out_features)
        # self.gaussian_mean = gaussian_mean
        # self.gaussian_std = gaussian_std

    def forward(self, H, A: Graph, normal_mask):
        """
        Main generation path used during training only
        """
        # 1. Find the index of labeled-normal node embeddings.
        #   Sample a subset of normal nodes for outlier generation
        n_idx = torch.where(normal_mask)[0]
        S = max(1, int(len(n_idx) * self.outlier_ratio))
        # Select S number of node id's from the normal nodes
        s_idx = n_idx[torch.randperm(len(n_idx), device=H.device)[:S]]

        # Create one outlier for each of the selected normal nodes based on their ego network.
        outlier_embeddings = []
        for i in s_idx:
            # 2. Select neighbor embeddings of each selected normal nodes (M, n_neighbors, out_features)
            #   This is also known as the "ego network"
            neighbors_idx = list(A.get_neighbors(int(i.item()))) # indices of neighbors
            # Convert to tensor so it can be passed to the model
            neighbors_idx = torch.tensor(neighbors_idx, dtype=torch.long, device=H.device)

            # 3. Pass those neighbor summaries through a learnable transform
            outlier_embed = F.relu(self.transform(H[neighbors_idx])).mean(dim=0)
            outlier_embeddings.append(outlier_embed)
        
        # Return all generated outliers as a tensor
        return torch.stack(outlier_embeddings), s_idx # (M, out_features), (M,)

# ================
# Loss functions
# ================
def compute_asymmetric_local_affinity(H_nodes, H_neighbors, neighbors_mask):
    """
    compute local affinity between all nodes and their neighbors
    outliers should have a lower affinity to neighbors than normal nodes

    Inputs:
        H_nodes: [M, out_features], M is number of M selected nodes
        H_neighbors: [M, n_neighbors, out_features], n_neighbors is number of neighbors for each node
        neighbors_mask: [M, n_neighbors], mask for valid neighbors (i.e. not generated outliers)
    Returns:
        affinity: [M, 1], affinity between each node and its neighbors
    """
    sims = F.cosine_similarity(
        H_nodes.unsqueeze(1), # (M, 1, d) -- i.e. only 1 neighbor (itself)
        H_neighbors, # (M, K, d)
        dim=-1 # average over all neighbors
    ) * neighbors_mask
    
    # average over all neighbors of the selected nodes
    return (sims.sum(dim=1) / neighbors_mask.sum(dim=1).clamp_min(1)).mean()

def compute_egocentric_closeness_loss(H_outlier, H_normal, epsilon):
    """
    compute L2 distance between generated outliers and their corresponding normal nodes 
    they were generated from (should be close to normal)
    """
    # Epsilon needs to be broadcasted to the same shape as H_normal
    epsilon = torch.full_like(H_normal, epsilon)
    return ((H_outlier - (H_normal + epsilon)) ** 2).sum(dim=1).mean()

def compute_loss(H_normal, H_normal_neighbors, H_outlier, H_outlier_neighbors, normal_neighbors_mask, outlier_neighbors_mask, ec_alpha, beta, gamma, bce_loss, epsilon):
    """
    Computes the two GGAD outlier losses from the paper.
    
    Inputs:
        H_normal: [N, out_features], N is number of normal nodes
        H_outlier: [S, out_features], S is number of generated outliers
        H_normal_neighbors: [S, n_neighbors, out_features]
        H_outlier_neighbors: [S, n_neighbors, out_features]
        normal_neighbors_mask, outlier_neighbors_mask: [S, n_neighbors] mask for valid neighbors/outliers
        bce_loss: binary cross-entropy loss from the classifier/discriminator
        ec_alpha, gamma, beta: hyperparameters for the loss
    Returns:
        affinity_loss, ec_loss, bce_loss, total_prior_loss
    """
    # compute affinities
    normal_affinity = compute_asymmetric_local_affinity(H_normal, H_normal_neighbors, normal_neighbors_mask)
    outlier_affinity = compute_asymmetric_local_affinity(H_outlier, H_outlier_neighbors, outlier_neighbors_mask)
    
    # compute losses
    ec_loss = compute_egocentric_closeness_loss(H_outlier, H_normal, epsilon)
    affinity_loss = torch.relu(ec_alpha - (normal_affinity - outlier_affinity))
    bce_loss = bce_loss # this is implemented by the classifier/discriminator

    total_prior_loss = bce_loss + (beta * affinity_loss) + (gamma * ec_loss)

    # return all for loss logging
    return affinity_loss, ec_loss, bce_loss, total_prior_loss