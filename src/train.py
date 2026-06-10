"""
As per the paper: "The goal of semi-supervised GAD is to learn an anomaly scoring function."
Validation: model can label anomalies by finding scores that are greater than a threshold without ground truth labels for anomalies.
Training: subset of labeled normal and outlier movies are fed into the model and trained to find the optimal scoring function.

To run: 
    python src/train.py
"""
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig
from hydra import initialize, compose

from src.gnc import GCNEncoder
from src.outlier_generator import OutlierGenerator, compute_loss

from src.utils import resolve_device

class AnomalyClassifier(nn.Module):
    """
    Train classifier to separate anomalies from normal movies based on embeddings.

    Input:
        H: [N, out_features], N is number of movies
    Output:
        y: [N, 1], label predictions as logits
    """
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1) # return 1 class (anomaly, normal)

    def forward(self, H):
        return self.fc(H).squeeze(1) # (N, 1) -> (N,)

class GGAD(nn.Module):
    """
    Wrapper module for encoder, outlier generator, and classifier.
    """
    def __init__(self, cfg, device):
        super().__init__()
        self.encoder = GCNEncoder(
            cfg.model.in_features,
            cfg.model.hidden_features,
            cfg.model.out_features,
            device,
        )
        self.outlier_generator = OutlierGenerator(
            cfg.model.out_features,
            device,
            cfg.ggad.generated_outlier_ratio,
            cfg.ggad.gaussian_mean,
            cfg.ggad.gaussian_std,
        )
        self.classifier = AnomalyClassifier(
            cfg.model.out_features
        )
    def forward(self, X, edge_index, graph, normal_mask):
        # Encode all movies/nodes from relationships (edges) and features (X)
        H = self.encoder(edge_index, X)

        # Generate outliers from normal movies
        H_outlier, s_idx = self.outlier_generator(H, graph, normal_mask)

        # Mask out the outliers from the normal movies
        H_normal = H[s_idx]
        
        # Get neighbors for each selected node
        neighbor_idx = [
            torch.tensor(graph.get_neighbors(int(i.item())), dtype=torch.long, device=H.device)
            for i in s_idx
        ]
        # Pad empty space in the batch dimension for nodes with no neighbors
        H_outlier_neighbors = pad_sequence([H[idx] for idx in neighbor_idx], batch_first=True)
        outlier_neighbors_mask = pad_sequence([torch.ones(len(idx), device=H.device) for idx in neighbor_idx], batch_first=True)

        # Return classifier score
        logits = self.classifier(torch.cat([H_normal, H_outlier], dim=0))
        labels = torch.cat([
            torch.ones(len(H_normal), device=H.device), # label 1 = normal
            torch.zeros(len(H_outlier), device=H.device)], # label 0 = outlier
        )

        return H, H_normal, H_normal_neighbors, H_outlier, H_outlier_neighbors, normal_neighbors_mask, outlier_neighbors_mask, s_idx, logits, labels



def train(cfg, X, edge_index, graph, normal_mask, train_loader, val_loader):
    # Basics
    device = resolve_device(cfg.device)
    wandb.init(project="sleeper-cinema-ggad", config=dict(cfg))

    # Model
    model = GGAD(cfg, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    bce_criterion = nn.BCEWithLogitsLoss()
    Path(cfg.paths.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    best_val_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} starting")
        train_loss = 0
        val_loss = 0
        
        # Train loop
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Pass through model
            H, H_normal, H_normal_neighbors, H_outlier, H_outlier_neighbors, normal_neighbors_mask, outlier_neighbors_mask, s_idx, logits, labels = model(
                X, edge_index, graph, normal_mask
            )

            # Compute losses
            bce_loss = bce_criterion(logits, labels)
            affinity_loss, ec_loss, bce_loss, total_prior_loss = compute_loss(
                H_normal, 
                H_normal_neighbors, 
                H_outlier, 
                H_outlier_neighbors, 
                normal_neighbors_mask, 
                outlier_neighbors_mask, 
                cfg.ggad.affinity_margin_alpha, 
                cfg.ggad.beta, 
                cfg.ggad["lambda"], 
                bce_loss, 
                cfg.ggad.epsilon
            )
            train_loss += total_prior_loss.item()

            # Backpropagate
            total_prior_loss.backward()
            optimizer.step()

            # Log losses
            wandb.log({
                "train/bce_loss": bce_loss.item(),
                "train/affinity_loss": affinity_loss.item(),
                "train/ec_loss": ec_loss.item(),
                "train/total_prior_loss": total_prior_loss.item(),
            })
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Pass through model
                H, H_normal, H_normal_neighbors, H_outlier, H_outlier_neighbors, normal_neighbors_mask, outlier_neighbors_mask, s_idx, logits, labels = model(
                    X, edge_index, graph, normal_mask
                )
                
                # Compute losses
                bce_loss = bce_criterion(logits, labels)
                affinity_loss, ec_loss, bce_loss, total_prior_loss = compute_loss(
                    H_normal, H_normal_neighbors, H_outlier, H_outlier_neighbors, normal_neighbors_mask, outlier_neighbors_mask, cfg.ggad.affinity_margin_alpha, cfg.ggad.beta, cfg.ggad["lambda"], bce_loss, cfg.ggad.epsilon
                )
                val_loss += total_prior_loss.item()
                
                # Log losses
                wandb.log({
                    "val/bce_loss": bce_loss.item(),
                    "val/affinity_loss": affinity_loss.item(),
                    "val/ec_loss": ec_loss.item(),
                    "val/total_prior_loss": total_prior_loss.item(),
                })

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} complete train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }

        # save latest model weights
        torch.save(checkpoint_payload, Path(cfg.paths.checkpoint_dir) / cfg.paths.latest_model)

        # save best model by validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_payload, Path(cfg.paths.checkpoint_dir) / cfg.paths.best_model)

        # collect checkpoints at the end of each epoch
        torch.save(checkpoint_payload, Path(cfg.paths.checkpoint_dir) / cfg.paths.epoch_model_pattern.format(epoch=epoch))

        # log epoch metrics to wandb
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "batch_size": cfg.training.batch_size,
            "epoch": epoch,
        })