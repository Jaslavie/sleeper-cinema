"""
Generates and adds outlier movies from film embeddings (feature representation space) 
of normal movies onto graph.

These are used as negative (synthetic) examples during training and is not used during inference.
"""
import torch.nn as nn
from gnc import GCNEncoder

class OutlierGenerator(nn.Module):
    """
    Generates pseudo-anomaly embeddings from labeled-normal film embeddings.
    Note that real anomalies (labeled as 1) are completely held out during this process.

    Inputs expected at training time:
        H_normal: encoded film embeddings from the GCN, shape [N, out_features]
        A: graph adjacency matrix, aligned with H
        normal_mask: boolean/index mask for normal films

    Output:
        H_outlier: generated outlier embeddings, shape [S, out_features], S = number of outliers
    """

    def __init__(
        self, 
        in_features, 
        hidden_features, 
        out_features, 
        device, 
        outlier_ratio, 
        gaussian_mean, 
        gaussian_std
    ):
        super().__init__()
        self.encoder = GCNEncoder(in_features, hidden_features, out_features, device)

    def forward(self, H, A, normal_mask):
        """
        Main generation path used during training only.
        """
        # 1. Select labeled-normal nodes
        normal_nodes = H[normal_mask]
        # 2. Compute each selected node's averaged neighbor embedding
        neighbor_embeddings = self.average_neighbor_embeddings(H, A, normal_nodes)
        # 3. Pass those neighbor summaries through a learnable transform
        outlier_embeddings = self.transform_neighbor_context(neighbor_embeddings)
        # 4. Add small Gaussian perturbation for egocentric closeness
        outlier_embeddings = self.add_gaussian_perturbation(outlier_embeddings)
        
        return outlier_embeddings

    def average_neighbor_embeddings(self, H, A, selected_nodes):
        """
        For each selected normal node, average the embeddings of its graph
        neighbors. This gives the local context that the outlier starts from.
        """
        raise NotImplementedError

    def transform_neighbor_context(self, neighbor_embeddings):
        """
        Apply the learnable outlier transform from the paper.

        This is the part that lets generated outliers adapt during training
        instead of being fixed random noise.
        """
        raise NotImplementedError

    def add_gaussian_perturbation(self, outlier_embeddings):
        """
        Add small Gaussian noise so generated outliers stay close to normal
        embeddings without becoming identical to them.
        """
        raise NotImplementedError

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
    sims = cosine_similarity(
        H_nodes.unsqueeze(1), # (M, 1, d) -- i.e. only 1 neighbor (itself)
        H_neighbors, # (M, K, d)
        dim=1 # average over all neighbors
    ) * neighbors_mask
    
    # average over all nodes
    return (1 / H_nodes.shape[0]) * torch.sum(sims)

def compute_egocentric_closeness_loss(H_outlier, H_normal):
    """
    compute L2 distance between generated outliers and their corresponding normal nodes 
    they were generated from (should be close to normal)
    """
    return (1 / n_outlier) * torch.sum((H_outlier - H_normal) ** 2).sum(dim=1).sum()

def compute_bce_loss(H_normal, H_outlier):
    """
    compute binary cross-entropy between normal and outlier nodes
    """
    # TODO
    pass

def compute_loss(H_normal, H_normal_neighbors, H_outlier, H_outlier_neighbors, ec_alpha, beta, gamma):
    """
    Computes the two GGAD outlier losses from the paper.
    
    Inputs:
        H_normal: [N, out_features], N is number of normal nodes
        H_outlier: [S, out_features], S is number of generated outliers
        H_normal_neighbors: [S, n_neighbors, out_features]
        H_outlier_neighbors: [S, n_neighbors, out_features]
        ec_alpha, gamma, beta: hyperparameters for the loss
    Returns:
        affinity_loss, ec_loss, bce_loss, total_prior_loss
    """
    n_outlier = H_outlier.shape[0]

    # compute affinities
    normal_affinity = compute_asymmetric_local_affinity(H_normal, H_normal_neighbors, neighbors_mask)
    outlier_affinity = compute_asymmetric_local_affinity(H_outlier, H_outlier_neighbors, neighbors_mask)
    
    # compute losses
    ec_loss = compute_egocentric_closeness_loss(H_outlier, H_normal)
    affinity_loss = max(0, ec_alpha - (normal_affinity - outlier_affinity))
    bce_loss = compute_bce_loss(H_normal, H_outlier)

    total_prior_loss = bce_loss + (beta * affinity_loss) + (gamma * ec_loss)

    # return all for loss logging
    return affinity_loss, ec_loss, bce_loss, total_prior_loss