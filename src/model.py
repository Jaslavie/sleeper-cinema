"""
Runs the Conditioned Random Field (CRF) model.

A gradient boosted tree assigns unary potential (i.e. the sleeper score of a 
film without considering its neighbors) to each film.

The mean field is a type of belief propagation algorithm that is used to 
approximate each film's score based on the interacting beliefs of its neighbors.
"""
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import HistGradientBoostingClassifier

from src.evaluate import auroc, auprc
from src.utils import load_graph, load_processed_movie_data, resolve_device


def mean_field(unary, neighbors, beta, known_idx, known_labels, iters=30):
    """
    The mean field forces the each film node to reconsider it's unary score by
    checking the beliefs of its neighbors. It is then adds a beta weight to pull
    the film's belief closer to the belief of its neighbors.

    Input:
        unary: Unary score of each film
        neighbors: Neighbors of each film
        beta: Beta weight for the mean field
        known_idx: Index of the known labels (these are strict and cannot be changed)
        known_labels: Labels of the known labels
        iters: Number of iterations for the mean field

    Output:
        P(sleeper) of each film
    """
    # Initialize the belief of each film to its unary score
    q = 1 / (1 + np.exp(-unary))
    q[known_idx] = known_labels

    # Iterate over number of rounds. This accounts for changes in beliefs
    # of each neighbor over time.
    for _ in range(iters):
        # Compute and add pull from neighbors to the belief of each film
        pull = np.array([beta * (2 * q[n].mean() - 1) if len(n) else 0.0 for n in neighbors])
        q = 1 / (1 + np.exp(-(unary + pull)))
        
        # Backfill the known labels
        q[known_idx] = known_labels
    
    return q


def run(cfg):
    device = resolve_device(cfg.device)

    # Load movie with mask over post-release features
    df, X, label = load_processed_movie_data(cfg.paths.csv_path, device)
    X = X.cpu().numpy() # tensor of data samples
    y = label.cpu().numpy().astype(int) # tensor of labels for each sample
    
    # Train and validation split
    n = len(df)
    train_size = int((1 - cfg.crf.val_frac) * n)
    train_idx = np.arange(train_size)
    val_idx = np.arange(train_size, n)

    # Create graph
    graph = load_graph(cfg.paths.graph_path)
    neighbors = [[j for j in graph.neighbors[i] if j != i] for i in range(n)]

    # Construct unary potential for each film
    # This predicts the anomaly score based purely on a film's pre-release features
    # i.e the film's individual belief about its own sleepiness
    gbm = HistGradientBoostingClassifier(
        learning_rate=cfg.crf.gbm_lr, max_leaf_nodes=cfg.crf.gbm_leaves,
        max_iter=cfg.crf.gbm_iters, l2_regularization=cfg.crf.gbm_l2,
    ).fit(X[train_idx], y[train_idx])

    # Compute unary score from the GBM
    p_unary = np.clip(gbm.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    unary = np.log(p_unary / (1 - p_unary))

    # The mean field adds context from the neighbors of the film to the unary score
    posterior = p_unary if cfg.crf.mrf_beta == 0 else mean_field(
        unary, neighbors, cfg.crf.mrf_beta, train_idx, y[train_idx], iters=cfg.crf.mrf_iters
    )

    # Evaluate unary-only vs mean field scores on validation films
    yv, pu, pp = y[val_idx], p_unary[val_idx], posterior[val_idx]
    metrics = dict(
        unary_auroc=auroc(yv, pu), unary_auprc=auprc(yv, pu),
        auroc=auroc(yv, pp), auprc=auprc(yv, pp), val_base_rate=float(yv.mean()),
    )
    print(f"val n={len(val_idx)} base={metrics['val_base_rate']:.3f}")
    print(f"unary: AUROC {metrics['unary_auroc']:.4f} AUPRC {metrics['unary_auprc']:.4f}")
    print(f"+mrf : AUROC {metrics['auroc']:.4f} AUPRC {metrics['auprc']:.4f}")

    # Save films ranked by final sleeper probability
    # We only care about the validation set
    out = Path(cfg.paths.artifacts_dir) / "crf_ranked.csv"
    out.parent.mkdir(exist_ok=True, parents=True)
    ranked = df.assign(sleeper_prob=posterior, split=np.where(np.arange(n) < train_size, "train", "val"))
    cols = [c for c in ["Release Group", "Success", "split", "sleeper_prob"] if c in ranked]
    ranked[ranked.split=='val'].sort_values("sleeper_prob", ascending=False)[cols].to_csv(out, index=False)
    
    return metrics

def main():
    cfg = OmegaConf.load("config/model.yaml")
    run(cfg)

if __name__ == "__main__":
    main()
