import torch
import torch.nn as nn
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def compute_anomaly_scores(model: nn.Module, X, edge_index) -> np.ndarray:
    """
    Args:
        model:      Trained GGAD model
        X:          [N, in_features] feature tensor
        edge_index: [2, E] edge index tensor
        device:     torch.device
    Output:
        anomaly_scores: np.ndarray of shape [N]
    """
    model.eval()
    with torch.no_grad():
        H = model.encoder(edge_index, X)
        full_logits = model.classifier(H).squeeze(-1)

    return (-full_logits).detach().cpu().numpy()

def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return roc_auc_score(y_true, scores)

def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return average_precision_score(y_true, scores)

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    top_k_indices = np.argsort(scores)[::-1][:k]
    return y_true[top_k_indices].mean()

def _compute_metrics(y_true: np.ndarray, scores: np.ndarray, k_values: list[int]) -> dict:
    anomaly_rate = y_true.mean()
    val_auroc = auroc(y_true, scores)
    val_auprc = auprc(y_true, scores)
    metrics = {"auroc": val_auroc, "auprc": val_auprc, "anomaly_rate": anomaly_rate}
    for k in k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
    return metrics

def run_evaluation(model, X, edge_index, df, val_indices, k_values = [10, 50, 100], save_csv = True) -> dict:
    """
    Args:
        model:       Trained GGAD model
        X:           [N, in_features] feature tensor
        edge_index:  [2, E] edge index tensor
        df:          Movie DataFrame with a 'Success' column
        val_indices: 
        k_values:    K values for Precision@K
        save_csv:    Whether to write ranked_movies.csv

    Returns:
        dict with keys: auroc, auprc, precision_at_{k} for each k
    """
    scores = compute_anomaly_scores(model, X, edge_index)
    y_true = df["Success"].astype(int).to_numpy()
    val_scores = scores[val_indices]
    val_y_true = y_true[val_indices]

    val_auroc = auroc(val_y_true, val_scores)
    val_auprc = auprc(val_y_true, val_scores)

    metrics = {"auroc": val_auroc, "auprc": val_auprc}
    for k in k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(val_y_true, val_scores, k)

    return metrics

def evaluate_final(model, X, edge_index, df, cfg, device) -> dict:
    """
    Args:
        model:      GGAD model instance (will be overwritten with best.pt weights)
        X:          [N, in_features] feature tensor
        edge_index: [2, E] edge index tensor
        df:         Movie DataFrame with a 'Success' column
        cfg:        Hydra config
        device:     torch.device
 
    Returns:
        dict with keys: auroc, auprc, precision_at_{k} for each k, sanity checks
    """
    best_path = Path(cfg.paths.checkpoint_dir) / cfg.paths.best_model
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch'] + 1} (val_loss={checkpoint['val_loss']:.4f})")
 
    scores = compute_anomaly_scores(model, X, edge_index)
    y_true = df["Success"].astype(int).to_numpy()
    k_values = [10, 50, 100]
    metrics = _compute_metrics(y_true, scores, k_values)
    anomaly_rate = metrics["anomaly_rate"]
 
    ranked_df = df.copy()
    ranked_df["anomaly_score"] = scores
 
    gross_col = "$Worldwide"
    if gross_col in ranked_df.columns:
        blockbuster_threshold = ranked_df[gross_col].quantile(0.90)
        blockbusters = ranked_df[ranked_df[gross_col] >= blockbuster_threshold]
        blockbuster_mean_score = blockbusters["anomaly_score"].mean()
        overall_mean_score = ranked_df["anomaly_score"].mean()
        metrics["sanity/blockbuster_mean_score"] = blockbuster_mean_score
        metrics["sanity/overall_mean_score"] = overall_mean_score
        metrics["sanity/blockbusters_score_below_mean"] = float(blockbuster_mean_score < overall_mean_score)
 
    sleeper_tail_threshold = np.percentile(scores, 80)
    sleeper_scores = scores[y_true == 1]
    metrics["sanity/sleeper_pct_in_top20"] = float((sleeper_scores >= sleeper_tail_threshold).mean())
 
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
 
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC curve (AUROC={metrics['auroc']:.3f})")
 
    axes[1].plot(rec, prec)
    axes[1].axhline(y=anomaly_rate, linestyle="--", color="gray", label=f"Baseline ({anomaly_rate:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-recall (AUPRC={metrics['auprc']:.3f})")
    axes[1].legend()
 
    axes[2].bar(k_values, [metrics[f"precision_at_{k}"] for k in k_values])
    axes[2].set_xticks(k_values)
    axes[2].axhline(y=anomaly_rate, linestyle="--", color="gray", label=f"Baseline ({anomaly_rate:.3f})")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision@K")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
 
    fig.suptitle(f"Final evaluation (best checkpoint epoch {checkpoint['epoch'] + 1})")
    plt.tight_layout()
 
    fig_path = Path(cfg.paths.checkpoint_dir) / "metrics_final.png"
    plt.savefig(fig_path)
    plt.close(fig)
    wandb.log({"eval/final_metrics_plot": wandb.Image(str(fig_path))})
 
    ranked_df["rank"] = pd.Series(scores).rank(ascending=False).astype(int).values
    ranked_df = ranked_df.sort_values("anomaly_score", ascending=False)
    out_path = Path(cfg.paths.checkpoint_dir) / cfg.paths.ranked_output
    ranked_df.to_csv(out_path, index=False)
    print(f"Ranked output saved to: {out_path}")
 
    return metrics
