import warnings

import torch
import torch.nn as nn
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix


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

def choose_threshold(y_true: np.ndarray, scores: np.ndarray, target_precision: float = 0.85) -> float:
    """
    Maximises F1 over thresholds that non-trivially split the data.

    Tried precision-first (target >= 0.85): too strict, missed most real sleepers.
    Tried raw F1 maximisation: threshold fell below all scores, flagging everything.
    Current fix: restrict candidates to thresholds within the actual score range
    that flag between 1%-99% of films, then pick the best F1 among those.
    """

    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    p = precisions[:-1]
    r = recalls[:-1]

    denom = p + r
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_scores = np.where(denom > 0, 2 * p * r / denom, 0.0)

    score_min, score_max = scores.min(), scores.max()
    in_range = (thresholds >= score_min) & (thresholds <= score_max)

    pct_flagged = np.array([(scores >= t).mean() for t in thresholds])
    non_trivial = (pct_flagged >= 0.01) & (pct_flagged <= 0.99)

    valid = in_range & non_trivial

    if not valid.any():
        sleeper_rate = y_true.mean()
        return float(np.percentile(scores, 100 * (1 - sleeper_rate)))

    best_idx = np.argmax(np.where(valid, f1_scores, -1))
    return float(thresholds[best_idx])
 
def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false alarm rate among normal films
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # miss rate among real sleepers
 
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

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
    best_threshold = choose_threshold(y_true, scores)
    threshold_m = threshold_metrics(y_true, scores, best_threshold)
    for key, val in threshold_m.items():
        metrics[f"thresh/{key}"] = val
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


    best_threshold = choose_threshold(y_true, scores)
    threshold_m = threshold_metrics(y_true, scores, best_threshold)
    for key, val in threshold_m.items():
        metrics[f"thresh/{key}"] = val

    return metrics

def _run_sanity_checks(ranked_df: pd.DataFrame, scores: np.ndarray, y_true: np.ndarray) -> tuple[dict, list[str]]:
    metrics = {}
    warnings = []
    gross_col = "$Worldwide"
 
    # Check 1: Blockbusters should score BELOW average
    if gross_col in ranked_df.columns:
        blockbuster_threshold = ranked_df[gross_col].quantile(0.90)
        blockbusters = ranked_df[ranked_df[gross_col] >= blockbuster_threshold]
        blockbuster_mean_score = blockbusters["anomaly_score"].mean()
        overall_mean_score = ranked_df["anomaly_score"].mean()
 
        passes = bool(blockbuster_mean_score < overall_mean_score)
        metrics["sanity/blockbuster_mean_score"] = blockbuster_mean_score
        metrics["sanity/overall_mean_score"] = overall_mean_score
        metrics["sanity/blockbusters_below_mean"] = float(passes)
 
        if not passes:
            warnings.append(
                f"SANITY FAIL — Blockbusters score ABOVE average "
                f"({blockbuster_mean_score:.3f} vs mean {overall_mean_score:.3f}). "
                f"The model may be treating high box office as suspicious rather than expected."
            )
        else:
            print(
                f"Blockbuster check passed: blockbuster mean score {blockbuster_mean_score:.3f} "
                f"< overall mean {overall_mean_score:.3f}"
            )
 
    # Check 2: Real sleepers should appear in the top 20% of scores 
    sleeper_tail_threshold = np.percentile(scores, 80)
    sleeper_scores = scores[y_true == 1]
    sleeper_pct_in_top20 = float((sleeper_scores >= sleeper_tail_threshold).mean())
    metrics["sanity/sleeper_pct_in_top20"] = sleeper_pct_in_top20
 
    if sleeper_pct_in_top20 < 0.50:
        warnings.append(
            f"SANITY FAIL — Only {sleeper_pct_in_top20:.1%} of real sleepers fall in the "
            f"top-20% of scores (expected >50%). The model is missing many real sleepers."
        )
    else:
        print(f"Sleeper recall check passed: {sleeper_pct_in_top20:.1%} of sleepers in top-20% scores")
 
    return metrics, warnings

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
 
    best_threshold = metrics["thresh/threshold"]
    ranked_df["predicted_sleeper"] = (scores >= best_threshold).astype(int)

    sanity_metrics, warnings = _run_sanity_checks(ranked_df, scores, y_true)
    metrics.update(sanity_metrics)

    if warnings:
        print("\n" + "=" * 60)
        print("SANITY CHECK FAILURES:")
        for w in warnings:
            print(w)
        print("=" * 60 + "\n")
        metrics["sanity/num_failures"] = len(warnings)
        wandb.log({"sanity/warnings": "\n".join(warnings)})
    else:
        print("All sanity checks passed")
        metrics["sanity/num_failures"] = 0

    tp = metrics["thresh/tp"]
    fp = metrics["thresh/fp"]
    fn = metrics["thresh/fn"]
    tn = metrics["thresh/tn"]
    print(
        f"At best-F1 threshold ({best_threshold:.3f}):\n"
        f"  Precision : {metrics['thresh/precision']:.3f}  "
        f"— of {tp+fp} flagged films, {tp} were real sleepers, {fp} were false alarms\n"
        f"  Recall    : {metrics['thresh/recall']:.3f}  "
        f"— caught {tp} of {tp+fn} real sleepers, missed {fn}\n"
        f"  F1        : {metrics['thresh/f1']:.3f}\n"
        f"  FPR       : {metrics['thresh/fpr']:.3f}  "
        f"— {fp} normal films wrongly flagged out of {fp+tn} total normal films\n"
        f"  FNR       : {metrics['thresh/fnr']:.3f}  "
        f"— {fn} real sleepers missed out of {tp+fn} total real sleepers"
    )

    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
 
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
 
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
 
    cm = np.array([[tn, fp], [fn, tp]])
    im = axes[3].imshow(cm, cmap="Blues")
    axes[3].set_xticks([0, 1])
    axes[3].set_yticks([0, 1])
    axes[3].set_xticklabels(["Predicted: Normal", "Predicted: Sleeper"])
    axes[3].set_yticklabels(["Actual: Normal", "Actual: Sleeper"])
    for i in range(2):
        for j in range(2):
            label = {(0,0): f"TN\n{tn}", (0,1): f"FP\n{fp}\n(false alarm)", (1,0): f"FN\n{fn}\n(missed sleeper)", (1,1): f"TP\n{tp}"}[(i,j)]
            axes[3].text(j, i, label, ha="center", va="center", fontsize=9,
                         color="white" if cm[i,j] > cm.max()/2 else "black")
    axes[3].set_title(f"Confusion matrix\n(threshold={best_threshold:.3f})")
    plt.colorbar(im, ax=axes[3])

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
