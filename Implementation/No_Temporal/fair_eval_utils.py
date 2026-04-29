from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import HeteroData


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def fit_cluster_label_mapping(y_true: np.ndarray, y_cluster: np.ndarray, n_clusters: int = 2) -> dict:
    n_classes = max(n_clusters, int(y_true.max()) + 1)
    table = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for cluster, true in zip(y_cluster, y_true):
        table[int(cluster), int(true)] += 1
    rows, cols = linear_sum_assignment(-table)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def apply_mapping(y_pred_raw: np.ndarray, mapping: Optional[dict]) -> np.ndarray:
    if mapping is None:
        return y_pred_raw.astype(int)
    return np.array([mapping.get(int(x), int(x)) for x in y_pred_raw], dtype=int)


def split_metrics(
    y_true_all: np.ndarray,
    y_pred_raw_all: np.ndarray,
    split_all: np.ndarray,
    split_id: int,
    mapping: Optional[dict] = None,
) -> Dict:
    mask = (y_true_all >= 0) & (split_all == split_id)
    y_true = y_true_all[mask].astype(int)
    y_raw = y_pred_raw_all[mask].astype(int)

    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "macro_f1": None,
            "weighted_f1": None,
            "ari": None,
            "nmi": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_pred = apply_mapping(y_raw, mapping)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ari": adjusted_rand_score(y_true, y_raw),
        "nmi": normalized_mutual_info_score(y_true, y_raw),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_all_splits_common(
    y_true_all: np.ndarray,
    y_pred_raw_all: np.ndarray,
    split_all: np.ndarray,
    use_train_mapping: bool,
    n_clusters: int = 2,
) -> Tuple[Dict, Optional[dict]]:
    mapping = None
    if use_train_mapping:
        train_mask = (y_true_all >= 0) & (split_all == 0)
        if train_mask.sum() == 0:
            raise ValueError("No labelled training nodes found. Cannot learn train mapping.")
        mapping = fit_cluster_label_mapping(
            y_true=y_true_all[train_mask],
            y_cluster=y_pred_raw_all[train_mask],
            n_clusters=n_clusters,
        )

    metrics = {
        "train": split_metrics(y_true_all, y_pred_raw_all, split_all, 0, mapping),
        "val": split_metrics(y_true_all, y_pred_raw_all, split_all, 1, mapping),
        "test": split_metrics(y_true_all, y_pred_raw_all, split_all, 2, mapping),
    }
    return metrics, mapping


def build_user_graph(num_users: int, edge_index: torch.Tensor, x_dim: int = 256) -> HeteroData:
    graph = HeteroData()
    graph["user"].x = torch.zeros(num_users, x_dim)
    graph[("user", "follows", "user")].edge_index = edge_index
    return graph


def save_common_outputs(
    run_dir: str,
    model_name: str,
    user_ids,
    labels: torch.Tensor,
    split: torch.Tensor,
    raw_predictions: np.ndarray,
    final_metrics: Dict,
    mapping: Optional[dict],
    history: Dict,
    timing: Dict,
    model: torch.nn.Module,
    x_shared_path: str,
):
    os.makedirs(run_dir, exist_ok=True)

    labels_np = labels.cpu().numpy()
    split_np = split.cpu().numpy()
    pred_labels = apply_mapping(raw_predictions, mapping)
    split_name = {0: "train", 1: "val", 2: "test"}

    pred_df = pd.DataFrame({
        "user_id": user_ids,
        "split_id": split_np,
        "split": [split_name.get(int(s), "unknown") for s in split_np],
        "true_label": labels_np,
        "pred_raw": raw_predictions,
        "pred_label": pred_labels,
    })
    pred_path = os.path.join(run_dir, "predictions_all.csv")
    pred_df.to_csv(pred_path, index=False)

    # Loss history: all per-epoch losses with equal length.
    loss_cols = {k: v for k, v in history.items() if k in {
        "loss", "contrastive", "classification", "pseudo_label",
        "reconstruction", "clustering", "botdcgc_loss", "cacl_loss",
        "botdcgc_reconstruction", "botdcgc_contrastive", "botdcgc_clustering",
        "cacl_contrastive", "cacl_pseudo_label"
    }}
    if loss_cols:
        max_len = max(len(v) for v in loss_cols.values())
        loss_df = pd.DataFrame({"epoch": list(range(1, max_len + 1))})
        for k, v in loss_cols.items():
            if len(v) == max_len:
                loss_df[k] = v
        loss_path = os.path.join(run_dir, "loss_history.csv")
        loss_df.to_csv(loss_path, index=False)

    metric_keys = [k for k in history.keys() if k.startswith(("train_", "val_", "test_"))]
    if "logged_epochs" in history and metric_keys:
        metric_df = pd.DataFrame({"epoch": history["logged_epochs"]})
        for k in metric_keys:
            metric_df[k] = history[k]
        metric_path = os.path.join(run_dir, "metric_history.csv")
        metric_df.to_csv(metric_path, index=False)

    results = {
        "model_name": model_name,
        "final": make_json_safe(final_metrics),
        "history": make_json_safe(history),
        "cluster_mapping": make_json_safe(mapping),
        "timing": timing,
        "num_users": len(user_ids),
        "num_model_parameters": int(sum(p.numel() for p in model.parameters())),
        "x_shared_path": x_shared_path,
    }
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return {
        "predictions": pred_path,
        "results": results_path,
    }
