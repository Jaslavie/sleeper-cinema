"""
As per the paper: "The goal of semi-supervised GAD is to learn an anomaly scoring function."
Validation: model can label anomalies by finding scores that are greater than a threshold without ground truth labels for anomalies.
Training: subset of labeled normal and outlier movies are fed into the model and trained to find the optimal scoring function.

To run (GGAD ablation — shows the GCN encoder is vestigial for this task):
    PYTHONPATH=. python -m src.evals.ggad.train
"""
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig
from hydra import initialize, compose
import hydra
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evals.ggad.gnc import GCNEncoder
from src.evals.ggad.outlier_generator import OutlierGenerator, compute_loss

from src.utils import resolve_device, load_processed_movie_data, load_graph

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
        )
        self.classifier = AnomalyClassifier(
            cfg.model.out_features
        )
    def forward(self, X, edge_index, graph, normal_mask, anomaly_mask=None):
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
        # Note that the embed/mask will be the same for both normal and outlier neighbors
        H_outlier_neighbors = pad_sequence([H[idx] for idx in neighbor_idx], batch_first=True)
        outlier_neighbors_mask = pad_sequence([torch.ones(len(idx), device=H.device) for idx in neighbor_idx], batch_first=True)
        H_normal_neighbors = H_outlier_neighbors
        normal_neighbors_mask = outlier_neighbors_mask

        # Return classifier score
        clf_inputs = [H_normal, H_outlier]
        clf_labels = [
            torch.ones(len(H_normal), device=H.device), # label 1 = normal
            torch.zeros(len(H_outlier), device=H.device)] # label 0 = outlier

        # Supervision: real known sleepers (anomaly_mask) are added as label-0 examples so the
        # classifier is oriented toward where true anomalies sit, not just the synthetic outliers.
        if anomaly_mask is not None and anomaly_mask.any():
            clf_inputs.append(H[anomaly_mask])
            clf_labels.append(torch.zeros(int(anomaly_mask.sum()), device=H.device))

        logits = self.classifier(torch.cat(clf_inputs, dim=0))
        labels = torch.cat(clf_labels)

        return H, H_normal, H_normal_neighbors, H_outlier, H_outlier_neighbors, normal_neighbors_mask, outlier_neighbors_mask, s_idx, logits, labels

@hydra.main(version_base=None, config_path="../../../config", config_name="model")
def train(cfg: DictConfig):
    # Basics
    device = resolve_device(cfg.device)
    wandb.init(project="sleeper-cinema-ggad", config=dict(cfg))

    # Initialize data (post-release features are masked out; label is 1 for sleepers)
    df, X, label = load_processed_movie_data(cfg.paths.csv_path, device)
    normal_mask = label == 0
    graph = load_graph(cfg.paths.graph_path)
    edge_index = torch.tensor(graph.get_edge_index(), dtype=torch.long, device=device).t().contiguous() # transpose to (2, E)
    dataset = TensorDataset(torch.arange(len(df)))

    train_randomizer = torch.Generator().manual_seed(42)
    val_randomizer = torch.Generator().manual_seed(43)

    # Split training data by samples
    num_movies = len(dataset)
    train_size = int(0.80 * num_movies)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, num_movies))
    train_idx = torch.tensor(train_indices, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_indices, dtype=torch.long, device=device)
    train_normal_mask = torch.zeros_like(normal_mask)
    val_normal_mask = torch.zeros_like(normal_mask)
    train_normal_mask[train_idx] = normal_mask[train_idx]
    val_normal_mask[val_idx] = normal_mask[val_idx]

    # Known sleepers in the train split, used as supervised anomaly examples (held-out ones are never shown)
    train_anomaly_mask = torch.zeros_like(normal_mask)
    train_anomaly_mask[train_idx] = (label[train_idx] == 1)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, cfg.training.batch_size, shuffle=True, drop_last=True, generator=train_randomizer
    )
    val_loader = DataLoader(
        val_dataset, cfg.training.batch_size, shuffle=True, drop_last=True, generator=val_randomizer
    )

    print(f"training size: {train_size}, validation size: {num_movies - train_size}")

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

            # Pass through model (with supervised sleepers from the train split)
            H, H_normal, H_normal_neighbors, H_outlier, H_outlier_neighbors, normal_neighbors_mask, outlier_neighbors_mask, s_idx, logits, labels = model(
                X, edge_index, graph, train_normal_mask, train_anomaly_mask
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
                    X, edge_index, graph, val_normal_mask
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

        # Score every film and measure ranking quality on the held-out split
        with torch.no_grad():
            H = model.encoder(edge_index, X)
            anomaly_score = 1 - torch.sigmoid(model.classifier(H)) # high score = likely sleeper
            val_y_true = label[val_idx].cpu().numpy()
            val_y_score = anomaly_score[val_idx].cpu().numpy()
            val_auroc = roc_auc_score(val_y_true, val_y_score) if len(set(val_y_true)) > 1 else float("nan")
            val_auprc = average_precision_score(val_y_true, val_y_score) if len(set(val_y_true)) > 1 else float("nan")
            ranked = df[["Release Group", "Success"]].copy()
            ranked["anomaly_score"] = anomaly_score.cpu().numpy()
            ranked.sort_values("anomaly_score", ascending=False).to_csv(Path(cfg.paths.checkpoint_dir) / cfg.paths.ranked_output, index=False)

        print(f"Epoch {epoch + 1}/{cfg.training.epochs} complete train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_auroc={val_auroc:.4f} val_auprc={val_auprc:.4f}")
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
            "val/anomaly_auroc": val_auroc,
            "val/anomaly_auprc": val_auprc,
            "batch_size": cfg.training.batch_size,
            "epoch": epoch,
        })

if __name__ == "__main__":
    train()
