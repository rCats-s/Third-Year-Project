from __future__ import annotations

import gc
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import normalize
from CL_model import adaptive_augment
from utils import drop_edge_weighted, pr_drop_weights
from copy import deepcopy
from torch_geometric.data import HeteroData
from sklearn.metrics import f1_score
from cacl_baseline import CACLBaseline, FeatureEncoderCfg, CACLCfg
from CL_model import adaptive_augment
from fair_eval_utils import (
    build_user_graph,
    evaluate_all_splits_common,
    save_common_outputs,
)


DATA_PATH    = r"D:\3rd year project\Implementation\Datasets\twibot22_out2\graph_data_model_ready_notemp_real.pt"
X_SHARED_PATH = r"D:\3rd year project\Implementation\runs\botdcgc_snowball_10k_notemp\x_shared.pt"
RUN_DIR      = r"D:\3rd year project\Implementation\runs\cacl_baseline_fair_notemp"

EPOCHS        = 200
LOG_EVERY     = 5
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
CL_BATCH_SIZE = 512    # nodes per mini-batch for contrastive loss (was full 7957 → OOM)
PATIENCE      = 30     # early stopping: logged evals with no val_f1 improvement

def _drop_edges(graph: HeteroData, drop_rate: float = 0.2) -> HeteroData:
    """Randomly remove a fraction of edges from each edge type."""
    g = deepcopy(graph)
    for edge_type in g.edge_types:
        ei = g[edge_type].edge_index          # (2, E)
        E  = ei.size(1)
        if E == 0:
            continue
        keep = torch.rand(E, device=ei.device) > drop_rate
        # Always keep at least one edge to avoid empty-graph errors
        if keep.sum() == 0:
            keep[0] = True
        g[edge_type].edge_index = ei[:, keep]
    return g


def minibatch_cl_loss(
    emb1: torch.Tensor,   # (N_train, D)
    emb2: torch.Tensor,   # (N_train, D)
    tau:  float = 0.5,
    batch_size: int = CL_BATCH_SIZE,
) -> torch.Tensor:
    """
    NT-Xent contrastive loss computed over random mini-batches of training nodes.

    For each mini-batch of `batch_size` nodes:
        refl_sim    = exp(sim(z1, z1) / tau)   (batch × batch)  — intra-view
        between_sim = exp(sim(z1, z2) / tau)   (batch × batch)  — inter-view
        loss_i = -log( between_sim[i,i] / (refl_sum[i] + between_sum[i] - refl[i,i]) )

    Memory:  batch × batch × 4 bytes × 4  =  512 × 512 × 16 = 4 MB per batch.
    Original: 7957 × 7957 × 16 = 1012 MB per call → OOM.
    """
    N = emb1.size(0)
    idx = torch.randperm(N, device=emb1.device)[:batch_size]

    z1 = normalize(emb1[idx], dim=1)   # (B, D)
    z2 = normalize(emb2[idx], dim=1)   # (B, D)

    refl_sim    = torch.exp(torch.mm(z1, z1.t()) / tau)   # (B, B)
    between_sim = torch.exp(torch.mm(z1, z2.t()) / tau)   # (B, B)

    loss1 = -torch.log(
        between_sim.diag() /
        (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() + 1e-8)
    )

    # Symmetric: also compute from view-2 perspective
    refl_sim2    = torch.exp(torch.mm(z2, z2.t()) / tau)
    between_sim2 = torch.exp(torch.mm(z2, z1.t()) / tau)
    loss2 = -torch.log(
        between_sim2.diag() /
        (refl_sim2.sum(1) + between_sim2.sum(1) - refl_sim2.diag() + 1e-8)
    )

    return ((loss1 + loss2) * 0.5).mean()


def cacl_forward_two_views_from_xshared(model: CACLBaseline, graph, X_shared):
    """
    Inject X_shared BEFORE augmentation.
    View 1: feature dropout only  (adaptive_augment default behaviour)
    View 2: edge dropout + feature dropout  (adds structural diversity)
    """
    g_with_x = model.inject_xshared(graph, X_shared)

    # Create structurally distinct graph for view 2 BEFORE adaptive_augment
    g_edge_dropped = _drop_edges(g_with_x, drop_rate=0.2)

    # adaptive_augment adds PageRank-weighted feature dropout on top
    aug_g1, _ = adaptive_augment(
        g_with_x,
        drop_feature_rate=model.cacl_cfg.drop_feature_rate,
    )
    aug_g2, _ = adaptive_augment(
        g_edge_dropped,
        drop_feature_rate=model.cacl_cfg.drop_feature_rate,
    )

    proj1, pred = model.cacl_model(aug_g1)
    proj2, _    = model.cacl_model(aug_g2)

    return {
        "proj1": proj1["user"],
        "proj2": proj2["user"],
        "pred":  pred,
    }


@torch.no_grad()
def predict_from_xshared(model: CACLBaseline, graph, X_shared):
    model.eval()
    g = model.inject_xshared(graph, X_shared)
    _, pred = model.cacl_model(g)
    probs = torch.softmax(pred, dim=1)[:, 1].cpu().numpy()
    return probs


def train_cacl_fixed_xshared(model, graph, X_shared, labels, split, device):
    model.to(device)
    X_shared = X_shared.to(device)
    labels   = labels.to(device)
    split    = split.to(device)
    graph    = graph.to(device)

    optimizer = optim.Adam(
        model.cacl_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    history = {
        "loss": [], "contrastive": [], "classification": [], "logged_epochs": [],
        "train_accuracy": [], "train_balanced_accuracy": [], "train_f1": [],
        "train_macro_f1": [], "train_ari": [], "train_nmi": [],
        "val_accuracy":   [], "val_balanced_accuracy":   [], "val_f1": [],
        "val_macro_f1":   [], "val_ari": [],               "val_nmi": [],
        "test_accuracy":  [], "test_balanced_accuracy":  [], "test_f1": [],
        "test_macro_f1":  [], "test_ari": [],              "test_nmi": [],
    }

    # Early stopping state
    best_val_f1       = -1.0
    best_epoch        = 0
    best_state        = None
    epochs_no_improve = 0

    print("=" * 60)
    print("CACL baseline fair run — fixed X_shared")
    print(f"epochs={EPOCHS}  lr={LR}  wd={WEIGHT_DECAY}")
    print(f"CL batch size: {CL_BATCH_SIZE} nodes  (was full {int((split==0).sum())} → OOM)")
    print(f"Early stopping patience: {PATIENCE} logged evals")
    print("=" * 60)

    train_mask = split == 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        fwd = cacl_forward_two_views_from_xshared(model, graph, X_shared)

        # Minibatch CL loss — only on training nodes, sampled subset
        train_proj1 = fwd["proj1"][train_mask]
        train_proj2 = fwd["proj2"][train_mask]
        l_cl = minibatch_cl_loss(train_proj1, train_proj2, tau=model.cacl_cfg.cl_tau)

        # Supervised classification loss on all training nodes
        train_labels = labels[train_mask]
        n_total = float(len(train_labels))
        n_bot   = float((train_labels == 1).sum().clamp(min=1))
        n_human = float((train_labels == 0).sum().clamp(min=1))
        class_weights = torch.tensor([n_total / (2 * n_human),
                                    n_total / (2 * n_bot)], device=device)
        l_cls = F.cross_entropy(fwd["pred"][train_mask], train_labels, weight=class_weights)

        loss = model.cacl_cfg.cl_alpha * l_cl + model.cacl_cfg.cl_beta * l_cls

        loss.backward()
        # Gradient clipping prevents loss spikes
        torch.nn.utils.clip_grad_norm_(model.cacl_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Explicit memory cleanup each epoch to prevent slow accumulation
        del fwd, train_proj1, train_proj2
        if ep % 10 == 0:
            gc.collect()

        history["loss"].append(loss.item())
        history["contrastive"].append(l_cl.item())
        history["classification"].append(l_cls.item())

        if ep % LOG_EVERY == 0 or ep == 1:
            raw_probs = predict_from_xshared(model, graph, X_shared)
            
            # Find best threshold on val for this checkpoint
            labels_np_loop = labels.detach().cpu().numpy()
            split_np_loop  = split.detach().cpu().numpy()
            val_mask_loop  = split_np_loop == 1
            
            thresholds = np.arange(0.05, 0.70, 0.01)
            best_t_loop = max(
                thresholds,
                key=lambda t: f1_score(
                    labels_np_loop[val_mask_loop],
                    raw_probs[val_mask_loop] > t,
                    zero_division=0,
                )
            )
            binary_pred = (raw_probs > best_t_loop).astype(int)
            
            metrics, _ = evaluate_all_splits_common(
                y_true_all=labels_np_loop,
                y_pred_raw_all=binary_pred,
                split_all=split_np_loop,
                use_train_mapping=False,
                n_clusters=2,
            )
            history["logged_epochs"].append(ep)
            for sname in ["train", "val", "test"]:
                history[f"{sname}_accuracy"].append(metrics[sname]["accuracy"])
                history[f"{sname}_balanced_accuracy"].append(metrics[sname]["balanced_accuracy"])
                history[f"{sname}_f1"].append(metrics[sname]["f1"])
                history[f"{sname}_macro_f1"].append(metrics[sname]["macro_f1"])
                history[f"{sname}_ari"].append(metrics[sname]["ari"])
                history[f"{sname}_nmi"].append(metrics[sname]["nmi"])

            current_val_f1 = metrics["val"]["f1"]
            marker = ""
            if current_val_f1 > best_val_f1:
                best_val_f1       = current_val_f1
                best_epoch        = ep
                best_state        = {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                marker = "  *** best ***"
            else:
                epochs_no_improve += 1

            print(
                f"epoch [{ep:03d}/{EPOCHS}] "
                f"loss={loss.item():.4f}  CL={l_cl.item():.4f}  cls={l_cls.item():.4f}  "
                f"train_f1={metrics['train']['f1']:.4f}  "
                f"val_f1={metrics['val']['f1']:.4f}  "
                f"test_f1={metrics['test']['f1']:.4f}"
                f"{marker}"
            )

            if epochs_no_improve >= PATIENCE:
                print(f"\n[early stop] epoch {ep}: no val_f1 improvement "
                      f"for {PATIENCE} evals. Best was epoch {best_epoch} "
                      f"val_f1={best_val_f1:.4f}")
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"\n[train] Restored best model from epoch {best_epoch} "
              f"(val_f1={best_val_f1:.4f})")

    raw_probs = predict_from_xshared(model, graph, X_shared)

    # Tune decision threshold on val set
    labels_np = labels.detach().cpu().numpy()
    split_np  = split.detach().cpu().numpy()
    val_mask_np = split_np == 1

    thresholds = np.arange(0.05, 0.70, 0.01)
    best_t = max(
        thresholds,
        key=lambda t: f1_score(
            labels_np[val_mask_np],
            raw_probs[val_mask_np] > t,
            zero_division=0,
        )
    )
    print(f"\n[threshold] Best val threshold: {best_t:.2f}")

    binary_pred = (raw_probs > best_t).astype(int)

    final, mapping = evaluate_all_splits_common(
        y_true_all=labels_np,
        y_pred_raw_all=binary_pred,
        split_all=split_np,
        use_train_mapping=False,
        n_clusters=2,
    )

    return {"history": history, "final": final, "raw_pred": binary_pred,
            "mapping": mapping, "threshold": float(best_t)}

def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()
    raw = torch.load(DATA_PATH, weights_only=False)
    X_shared = torch.load(X_SHARED_PATH, weights_only=False).float()
    load_x_time = time.time() - start

    labels = raw["labels"]
    split = raw["split"]
    edge_index = raw["edge_index"]
    user_ids = raw["user_ids"]

    graph = build_user_graph(len(user_ids), edge_index, x_dim=X_shared.shape[1])
    metadata = (graph.node_types, graph.edge_types)

    feat_cfg = FeatureEncoderCfg(embed_dim=X_shared.shape[1])
    cacl_cfg = CACLCfg(
        hetero_model="SAGE",
        hidden_channels=64,
        out_channels=16,
        proj_out=16,
        proj_hidden=32,
        num_layer=1,
    )
    model = CACLBaseline(
        metadata=metadata,
        in_channels={"user": X_shared.shape[1]},
        feat_cfg=feat_cfg,
        cacl_cfg=cacl_cfg,
        use_classifier=True,
    )

    train_start = time.time()
    result = train_cacl_fixed_xshared(model, graph, X_shared, labels, split, device)
    train_time = time.time() - train_start

    model_path = os.path.join(RUN_DIR, "cacl_baseline_model_notemp.pt")
    torch.save({
        "cacl_model": model.cacl_model.state_dict(),
        "model_config": {
            "x_shared_dim": int(X_shared.shape[1]),
            "hetero_model": cacl_cfg.hetero_model,
            "hidden_channels": cacl_cfg.hidden_channels,
            "out_channels": cacl_cfg.out_channels,
            "proj_out": cacl_cfg.proj_out,
            "proj_hidden": cacl_cfg.proj_hidden,
            "num_layer": cacl_cfg.num_layer,
        },
        "final_metrics": result["final"],
        "history": result["history"],
    }, model_path)

    save_common_outputs(
        run_dir=RUN_DIR,
        model_name="cacl_baseline_fixed_xshared_notemp",
        user_ids=user_ids,
        labels=labels,
        split=split,
        raw_predictions=result["raw_pred"],
        final_metrics=result["final"],
        mapping=result["mapping"],
        history=result["history"],
        timing={
            "x_shared_load_seconds": load_x_time,
            "training_and_evaluation_seconds": train_time,
            "total_experiment_seconds": load_x_time + train_time,
        },
        model=model,
        x_shared_path=X_SHARED_PATH,
    )
    print(f"Saved fair CACL run to: {RUN_DIR}")


if __name__ == "__main__":
    main()