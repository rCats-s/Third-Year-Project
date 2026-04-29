import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from tqdm import tqdm
from feature_encoder import UserFeatureEncoder
from graph_autoencoder import BotDCGCGraphAutoEncoder, reconstruction_loss
from clustering import ClusteringModule, clustering_loss, total_loss
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import balanced_accuracy_score


def _lap(cost):
    r, c = linear_sum_assignment(cost)
    return list(zip(r, c))

base_path = r"D:\3rd year project\Implementation\Datasets\twibot22_out2"

class BotDCGC(nn.Module):
    def __init__(
    self,
    user_embed_dim: int = 256,
    gat_hidden_dims=None,
    n_clusters: int = 2,
    order_t: int = 1,
    alpha: float = 0.2,
    tau: float = 1.0,
    gamma1: float = 2.0,
    gamma2: float = 10.0,
    undirected: bool = True,
    add_self_loops: bool = False,
    ):
        super().__init__()

        if gat_hidden_dims is None:
            gat_hidden_dims = [256, 256, 16]

        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.gae = BotDCGCGraphAutoEncoder(
            in_dim=user_embed_dim,
            gat_hidden_dims=gat_hidden_dims,
            order_t=order_t,
            alpha=alpha,
            tau=tau,
            undirected=undirected,
            add_self_loops=add_self_loops,
            dropout = 0.1,
        )

        self.clustering = ClusteringModule(
            n_clusters=n_clusters,
            embed_dim=gat_hidden_dims[-1],
        )

    def forward_from_shared_features(self, X_shared, edge_index):
        Z, A_hat, A, M = self.gae(X_shared, edge_index)
        Q, P = self.clustering(Z)

        return {
            "X": X_shared,
            "Z": Z,
            "A_hat": A_hat,
            "A": A,
            "M": M,
            "Q": Q,
            "P": P,
        }

    def compute_losses(self, outputs, recon_pos_weight=None):
        Z = outputs["Z"]
        A_hat = outputs["A_hat"]
        A = outputs["A"]
        M = outputs["M"]
        Q = outputs["Q"]
        P = outputs["P"]

        l_recon, l_contrast = self.gae.compute_losses(
            Z, A_hat, A, M, recon_pos_weight=recon_pos_weight
        )
        l_cluster = clustering_loss(P, Q)

        Q_mean = Q.mean(dim=0).clamp(min=1e-9)
        cluster_entropy = -(Q_mean * Q_mean.log()).sum()
        max_entropy = torch.log(torch.tensor(float(Q.size(1)), device=Q.device))
        l_entropy = max_entropy - cluster_entropy       # 0=balanced, grows when collapsed

        l_total = total_loss(
            l_recon=l_recon,
            l_contrast=l_contrast,
            l_cluster=l_cluster + 0.1 * l_entropy,
            gamma1=self.gamma1,
            gamma2=self.gamma2,
        )

        return {
            "loss": l_total,
            "reconstruction": l_recon,
            "contrastive": l_contrast,
            "clustering": l_cluster,
        }

    @torch.no_grad()
    def initialise_clusters_from_shared_features(self, X_shared, edge_index):
        self.eval()
        outputs = self.forward_from_shared_features(X_shared, edge_index)
        self.clustering.initialise(outputs["Z"])

    @torch.no_grad()
    def predict_from_shared_features(self, X_shared, edge_index):
        self.eval()
        outputs = self.forward_from_shared_features(X_shared, edge_index)
        return self.clustering.predict(outputs["Z"])
    

from scipy.optimize import linear_sum_assignment

def _lap(cost):
    r, c = linear_sum_assignment(cost)
    return list(zip(r, c))


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int = 2) -> Tuple[float, np.ndarray]:
    n_classes = max(n_clusters, int(y_true.max()) + 1)
    D = np.zeros((n_clusters, n_classes), dtype=np.int64)

    for pred, true in zip(y_pred, y_true):
        D[pred, true] += 1

    pairs = _lap(-D)
    mapping = {p: t for p, t in pairs}
    y_remap = np.array([mapping.get(p, p) for p in y_pred])

    acc = accuracy_score(y_true, y_remap)
    return acc, y_remap

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

def fit_cluster_label_mapping(
    y_true: np.ndarray,
    y_cluster: np.ndarray,
    n_clusters: int = 2,
) -> dict:
    """
    Learns cluster_id -> class_label mapping using labelled TRAIN nodes only.
    This avoids using test labels to decide what cluster 0/1 means.
    """
    n_classes = max(n_clusters, int(y_true.max()) + 1)
    D = np.zeros((n_clusters, n_classes), dtype=np.int64)

    for cluster, true in zip(y_cluster, y_true):
        D[cluster, true] += 1

    pairs = _lap(-D)
    mapping = {cluster_id: class_id for cluster_id, class_id in pairs}
    return mapping


def apply_cluster_mapping(y_cluster: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Converts predicted cluster IDs into predicted class labels.
    Unknown clusters fall back to their original ID.
    """
    return np.array([mapping.get(int(c), int(c)) for c in y_cluster])


@torch.no_grad()
def predict_clusters_from_xshared(model, X_shared, edge_index):
    model.eval()
    clusters = model.predict_from_shared_features(
        X_shared,
        edge_index,
    ).cpu().numpy()
    return clusters


def evaluate_with_fixed_mapping(
    y_true_all: np.ndarray,
    y_cluster_all: np.ndarray,
    split_all: np.ndarray,
    split_id: int,
    mapping: dict,
):
    """
    Evaluate one split using a cluster->label mapping learned from train.
    split_id: 0=train, 1=val, 2=test
    """
    mask = (y_true_all >= 0) & (split_all == split_id)

    y_true = y_true_all[mask]
    y_cluster = y_cluster_all[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    y_pred = apply_cluster_mapping(y_cluster, mapping)
    tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
    ).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ari": adjusted_rand_score(y_true, y_cluster),
        "nmi": normalized_mutual_info_score(y_true, y_cluster),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_all_splits(model, dataset, X_shared, edge_index):
    """
    Split-aware evaluation using fixed precomputed X_shared.
    Cluster mapping is learned from train split only.
    """
    y_cluster_all = predict_clusters_from_xshared(model, X_shared, edge_index)
    y_true_all = dataset.labels.cpu().numpy()
    split_all = dataset.split.cpu().numpy()

    train_mask = (y_true_all >= 0) & (split_all == 0)

    if train_mask.sum() == 0:
        raise ValueError("No labelled training nodes found. Cannot learn cluster mapping.")

    mapping = fit_cluster_label_mapping(
        y_true=y_true_all[train_mask],
        y_cluster=y_cluster_all[train_mask],
        n_clusters=model.clustering.K,
    )

    metrics = {
        "train": evaluate_with_fixed_mapping(y_true_all, y_cluster_all, split_all, 0, mapping),
        "val": evaluate_with_fixed_mapping(y_true_all, y_cluster_all, split_all, 1, mapping),
        "test": evaluate_with_fixed_mapping(y_true_all, y_cluster_all, split_all, 2, mapping),
    }

    return metrics, y_cluster_all, mapping
def encode_shared_features_batched(
    shared_feature_encoder,
    dataset,
    device,
    batch_size: int = 32,
):
    outputs = []
    N = dataset.num.size(0)

    for start in tqdm(range(0, N, batch_size), desc="  encoding X_shared"):
        end = min(start + batch_size, N)

        x = shared_feature_encoder(
            dataset.num[start:end].to(device),
            dataset.cat[start:end].to(device),
            dataset.desc_emb[start:end].to(device),
            dataset.tweet_emb[start:end].to(device),
            dataset.tweet_len[start:end].to(device),
        )

        outputs.append(x)

    return torch.cat(outputs, dim=0)
@torch.no_grad()
def encode_shared_features_batched_no_grad(
    shared_feature_encoder,
    dataset,
    device,
    batch_size: int = 32,
):
    shared_feature_encoder.eval()
    outputs = []
    N = dataset.num.size(0)

    for start in tqdm(range(0, N, batch_size), desc="  encoding X_shared"):
        end = min(start + batch_size, N)

        x = shared_feature_encoder(
            dataset.num[start:end].to(device),
            dataset.cat[start:end].to(device),
            dataset.desc_emb[start:end].to(device),
            dataset.tweet_emb[start:end].to(device),
            dataset.tweet_len[start:end].to(device),
        )

        outputs.append(x.detach().cpu())

    return torch.cat(outputs, dim=0)
def get_or_create_x_shared(
    shared_feature_encoder,
    dataset,
    device,
    x_shared_path: str,
    batch_size: int = 32,
):
    if dataset.x_shared is not None:
        print("[X_shared] Found x_shared inside dataset .pt")
        return dataset.x_shared.float()

    if x_shared_path and os.path.exists(x_shared_path):
        print(f"[X_shared] Loading precomputed X_shared from {x_shared_path}")
        return torch.load(x_shared_path, weights_only=False).float()

    print("[X_shared] Computing X_shared once...")
    x_shared = encode_shared_features_batched_no_grad(
        shared_feature_encoder,
        dataset,
        device,
        batch_size=batch_size,
    ).float()

    if x_shared_path:
        torch.save(x_shared, x_shared_path)
        print(f"[X_shared] Saved X_shared to {x_shared_path}")

    return x_shared
def train(
    model,
    dataset,
    X_shared,
    device,
    epochs: int = 200,
    lr: float = 1e-3,
    wd: float = 1e-3,
    pretrain_epochs: int = 30,
    log_every: int = 10,
    save_path: str = None,
    grad_clip: float = 1.0,
    cluster_refresh_every: int = 20,
    patience: int = 30,
):
    

    model.to(device)
    X_shared = X_shared.to(device)
    edge_index = dataset.edge_index.to(device)

    print("\n===== SANITY CHECK =====")
    print("X_shared shape :", X_shared.shape)
    print("edge_index shape:", edge_index.shape)

    assert X_shared.shape[0] == len(dataset.user_ids), (
        f"X_shared rows {X_shared.shape[0]} != users {len(dataset.user_ids)}"
    )
    assert X_shared.shape[1] == 256, (
        f"Expected X_shared dim 256, got {X_shared.shape[1]}"
    )

    print("X_shared has nan:", torch.isnan(X_shared).any().item())
    print("X_shared has inf:", torch.isinf(X_shared).any().item())

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    # Early stopping state
    best_val_f1    = -1.0
    best_epoch     = 0
    best_state     = None
    epochs_no_improve = 0

    print(f"[train] Pretraining graph autoencoder for {pretrain_epochs} epochs...")
    for ep in range(1, pretrain_epochs + 1):
        model.train()
        optimizer.zero_grad()

        outputs = model.forward_from_shared_features(X_shared, edge_index)
        l_recon = reconstruction_loss(outputs["A"], outputs["A_hat"])

        l_recon.backward()
        optimizer.step()

        if ep % 10 == 0 or ep == 1:
            print(f"  pretrain [{ep:03d}/{pretrain_epochs}]  L_recon={l_recon.item():.4f}")

    print("[train] Initialising cluster centres with K-means...")
    with torch.no_grad():
        model.initialise_clusters_from_shared_features(X_shared, edge_index)

    print(f"[train] Main training for {epochs} epochs...")

    history = {
        "loss": [],
        "reconstruction": [],
        "contrastive": [],
        "clustering": [],
        "logged_epochs": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_accuracy": [],
        "val_f1": [],
        "test_accuracy": [],
        "test_f1": [],
        "train_balanced_accuracy": [],
        "val_balanced_accuracy": [],
        "test_balanced_accuracy": [],

        "train_macro_f1": [],
        "val_macro_f1": [],
        "test_macro_f1": [],

        "train_ari": [],
        "val_ari": [],
        "test_ari": [],

        "train_nmi": [],
        "val_nmi": [],
        "test_nmi": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        outputs = model.forward_from_shared_features(X_shared, edge_index)
        losses = model.compute_losses(outputs)

        losses["loss"].backward()

        # Gradient clipping — prevents loss spikes from large gradient steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        # Periodic cluster centre refresh — re-anchors centres to current
        # embeddings so they don't drift apart as the GAT updates
        if ep % cluster_refresh_every == 0:
            with torch.no_grad():
                model.initialise_clusters_from_shared_features(X_shared, edge_index)

        for k, v in losses.items():
            history[k].append(v.item())

        if ep % log_every == 0 or ep == 1:
            split_metrics, _, _ = evaluate_all_splits(
                model,
                dataset,
                X_shared,
                edge_index,
            )

            history["logged_epochs"].append(ep)
            history["train_accuracy"].append(split_metrics["train"]["accuracy"])
            history["train_f1"].append(split_metrics["train"]["f1"])
            history["val_accuracy"].append(split_metrics["val"]["accuracy"])
            history["val_f1"].append(split_metrics["val"]["f1"])
            history["test_accuracy"].append(split_metrics["test"]["accuracy"])
            history["test_f1"].append(split_metrics["test"]["f1"])
            history["train_balanced_accuracy"].append(split_metrics["train"]["balanced_accuracy"])
            history["val_balanced_accuracy"].append(split_metrics["val"]["balanced_accuracy"])
            history["test_balanced_accuracy"].append(split_metrics["test"]["balanced_accuracy"])

            history["train_macro_f1"].append(split_metrics["train"]["macro_f1"])
            history["val_macro_f1"].append(split_metrics["val"]["macro_f1"])
            history["test_macro_f1"].append(split_metrics["test"]["macro_f1"])

            history["train_ari"].append(split_metrics["train"]["ari"])
            history["val_ari"].append(split_metrics["val"]["ari"])
            history["test_ari"].append(split_metrics["test"]["ari"])

            history["train_nmi"].append(split_metrics["train"]["nmi"])
            history["val_nmi"].append(split_metrics["val"]["nmi"])
            history["test_nmi"].append(split_metrics["test"]["nmi"])
            

            print(
                f"  epoch [{ep:03d}/{epochs}] "
                f"loss={losses['loss'].item():.4f} "
                f"rec={losses['reconstruction'].item():.4f} "
                f"con={losses['contrastive'].item():.4f} "
                f"clu={losses['clustering'].item():.4f} "
                f"train_f1={split_metrics['train']['f1']:.4f} "
                f"val_f1={split_metrics['val']['f1']:.4f} "
                f"test_f1={split_metrics['test']['f1']:.4f}"
            )

    final, final_clusters, final_mapping = evaluate_all_splits(
        model,
        dataset,
        X_shared,
        edge_index,
    )

    print("\n[train] Final split-aware metrics:")
    for split_name in ["train", "val", "test"]:
        print(f"\n  {split_name.upper()}:")
        for k, v in final[split_name].items():
            if isinstance(v, float):
                print(f"    {k:10s}: {v:.4f}")
            else:
                print(f"    {k:10s}: {v}")

    print("\n  Cluster mapping learned from train split:")
    print(f"    {final_mapping}")

    if save_path:
        torch.save({
            "botdcgc_branch": model.state_dict(),
            "model_config": {
                "user_embed_dim": 256,
                "gat_hidden_dims": [64, 16],
                "n_clusters": 2,
                "order_t": 1,
                "alpha": 0.2,
                "tau": 1.0,
                "gamma1": 2.0,
                "gamma2": 10.0,
            },
            "x_shared_shape": tuple(X_shared.shape),
            "final_metrics": final,
            "cluster_mapping": {int(k): int(v) for k, v in final_mapping.items()},
            "history": history,
        }, save_path)

        print(f"[train] Model saved to {save_path}")

    return {
        "history": history,
        "final": final,
        "final_clusters": final_clusters,
        "final_mapping": final_mapping,
    }

class SimpleDataset:
    def __init__(self, d):
        self.user_ids = d["user_ids"]
        self.labels = d["labels"]
        self.split = d["split"]
        self.edge_index = d["edge_index"]

        self.num = d["num"]
        self.cat = d["cat"]
        self.desc_emb = d["desc_emb"]
        self.tweet_emb = d["tweet_emb"]
        self.tweet_len = d["tweet_len"]

        self.x_shared = d.get("x_shared", None)


if __name__ == "__main__":
    import time
    import json

    run_dir = r"D:\3rd year project\Implementation\runs\botdcgc_snowball_10k_notemp"
    os.makedirs(run_dir, exist_ok=True)

    data_path = r"D:\3rd year project\Implementation\Datasets\twibot22_out2\graph_data_model_ready_notemp_real.pt"
    model_path = os.path.join(run_dir, "botdcgc_model_notemp.pt")
    x_shared_path = os.path.join(run_dir, "x_shared.pt")
    encoder_path = os.path.join(run_dir, "feature_encoder_initial.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_start_time = time.time()


    raw = torch.load(data_path, weights_only=False)
    dataset = SimpleDataset(raw)

    shared_feature_encoder = UserFeatureEncoder(
        num_features=7,
        cat_features=11,
        roberta_dim=768,
        lstm_hidden=128,
        embed_dim=256,
    ).to(device)

    # Save encoder state so X_shared is reproducible
    torch.save({
        "shared_feature_encoder": shared_feature_encoder.state_dict(),
        "encoder_config": {
            "num_features": 7,
            "cat_features": 11,
            "roberta_dim": 768,
            "lstm_hidden": 128,
            "embed_dim": 256,
        },
    }, encoder_path)

    X_shared = get_or_create_x_shared(
    shared_feature_encoder,
    dataset,
    device,
    x_shared_path=x_shared_path,
    batch_size=32,
    )
    x_shared_time = time.time() - x_start_time

    print("X_shared shape:", tuple(X_shared.shape))
    print(f"X_shared preparation time: {x_shared_time:.2f} seconds")


    model = BotDCGC(
        user_embed_dim=256,
        gat_hidden_dims=[64, 16],
        n_clusters=2,
        order_t=1,
        alpha=0.2,
        tau=1.0,
        gamma1=0.5,
        gamma2=5.0,
        undirected=True,
        add_self_loops=True,
    )

    botdcgc_start_time = time.time()

    results = train(
        model=model,
        dataset=dataset,
        X_shared=X_shared,
        device=device,
        epochs=200,
        pretrain_epochs=20,
        log_every=5,
        save_path=model_path,
        lr=1e-3,          # add if your train() function accepts it — was 5e-3
    )
    loss_df = pd.DataFrame({
    "epoch": list(range(1, len(results["history"]["loss"]) + 1)),
    "loss": results["history"]["loss"],
    "reconstruction": results["history"]["reconstruction"],
    "contrastive": results["history"]["contrastive"],
    "clustering": results["history"]["clustering"],
    })
    loss_path = os.path.join(run_dir, "loss_history.csv")
    loss_df.to_csv(loss_path, index=False)
    print(f"Saved loss history to {loss_path}")

    metric_df = pd.DataFrame({
        "epoch": results["history"]["logged_epochs"],

        "train_accuracy": results["history"]["train_accuracy"],
        "train_balanced_accuracy": results["history"]["train_balanced_accuracy"],
        "train_f1": results["history"]["train_f1"],
        "train_macro_f1": results["history"]["train_macro_f1"],
        "train_ari": results["history"]["train_ari"],
        "train_nmi": results["history"]["train_nmi"],

        "val_accuracy": results["history"]["val_accuracy"],
        "val_balanced_accuracy": results["history"]["val_balanced_accuracy"],
        "val_f1": results["history"]["val_f1"],
        "val_macro_f1": results["history"]["val_macro_f1"],
        "val_ari": results["history"]["val_ari"],
        "val_nmi": results["history"]["val_nmi"],

        "test_accuracy": results["history"]["test_accuracy"],
        "test_balanced_accuracy": results["history"]["test_balanced_accuracy"],
        "test_f1": results["history"]["test_f1"],
        "test_macro_f1": results["history"]["test_macro_f1"],
        "test_ari": results["history"]["test_ari"],
        "test_nmi": results["history"]["test_nmi"],
    })
    metric_path = os.path.join(run_dir, "metric_history.csv")
    metric_df.to_csv(metric_path, index=False)
    print(f"Saved metric history to {metric_path}")
    botdcgc_training_time = time.time() - botdcgc_start_time
    final_metrics, final_clusters, final_mapping = evaluate_all_splits(
        model,
        dataset,
        X_shared.to(device),
        dataset.edge_index.to(device),
    )

    labels_np = dataset.labels.cpu().numpy()
    split_np = dataset.split.cpu().numpy()
    pred_labels = apply_cluster_mapping(final_clusters, final_mapping)

    split_name = {0: "train", 1: "val", 2: "test"}

    pred_df = pd.DataFrame({
        "user_id": dataset.user_ids,
        "split_id": split_np,
        "split": [split_name.get(int(s), "unknown") for s in split_np],
        "true_label": labels_np,
        "pred_cluster": final_clusters,
        "pred_label": pred_labels,
    })

    pred_path = os.path.join(run_dir, "predictions_all.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    final_mapping_json = {str(k): int(v) for k, v in final_mapping.items()}

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(
            {
                "final": make_json_safe(final_metrics),
                "history": make_json_safe(results["history"]),
                "cluster_mapping": final_mapping_json,
                 "timing": {
                    "x_shared_preparation_seconds": x_shared_time,
                    "botdcgc_training_and_evaluation_seconds": botdcgc_training_time,
                    "total_experiment_seconds": x_shared_time + botdcgc_training_time,
                },
                "device": str(device),
                "num_users": len(dataset.user_ids),
                "num_edges": int(dataset.edge_index.shape[1]),
                "x_shared_path": x_shared_path,
                "num_model_parameters": sum(p.numel() for p in model.parameters()),
                
            },
            f,
            indent=2,
        )

    print(f"\nSaved experiment results to: {run_dir}")