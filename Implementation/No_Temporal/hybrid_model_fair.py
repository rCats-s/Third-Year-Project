from __future__ import annotations

import gc
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.nn.functional import normalize
from torch_geometric.data import HeteroData
from sklearn.metrics import f1_score
from CL_model import adaptive_augment
from hybrid_model import HybridModel, FeatureEncoderCfg, BotDCGCCfg, CACLCfg
from clustering import clustering_loss, total_loss as botdcgc_total_loss
from graph_autoencoder import reconstruction_loss
from fair_eval_utils import (
    build_user_graph,
    evaluate_all_splits_common,
    save_common_outputs,
)

DATA_PATH     = r"D:\3rd year project\Implementation\Datasets\twibot22_out2\graph_data_model_ready_notemp_real.pt"
X_SHARED_PATH = r"D:\3rd year project\Implementation\runs\botdcgc_snowball_10k_notemp\x_shared.pt"
RUN_DIR       = r"D:\3rd year project\Implementation\runs\hybrid_fair_notemp"

PRETRAIN_EPOCHS  = 20
BOTDCGC_EPOCHS   = 200
CACL_EPOCHS      = 200
LOG_EVERY        = 5
PATIENCE         = 30       # early stopping on val_f1 (CACL phase)

BOTDCGC_LR      = 1e-3
BOTDCGC_WD      = 1e-3
CACL_LR         = 1e-3
CACL_WD         = 1e-4
PSEUDO_CONF_THR = 0.80
CL_BATCH_SIZE   = 512       # same as CACL baseline — avoids OOM


# ── Helpers shared with CACL baseline ────────────────────────────────────────

def _drop_edges(graph: HeteroData, drop_rate: float = 0.2) -> HeteroData:
    """Randomly remove a fraction of edges from each edge type."""
    g = deepcopy(graph)
    for edge_type in g.edge_types:
        ei = g[edge_type].edge_index
        E  = ei.size(1)
        if E == 0:
            continue
        keep = torch.rand(E, device=ei.device) > drop_rate
        if keep.sum() == 0:
            keep[0] = True
        g[edge_type].edge_index = ei[:, keep]
    return g


def minibatch_cl_loss(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    tau: float = 0.5,
    batch_size: int = CL_BATCH_SIZE,
) -> torch.Tensor:
    """NT-Xent loss over a random mini-batch — identical to CACL baseline."""
    N   = emb1.size(0)
    idx = torch.randperm(N, device=emb1.device)[:batch_size]
    z1  = normalize(emb1[idx], dim=1)
    z2  = normalize(emb2[idx], dim=1)

    refl_sim    = torch.exp(torch.mm(z1, z1.t()) / tau)
    between_sim = torch.exp(torch.mm(z1, z2.t()) / tau)
    loss1 = -torch.log(
        between_sim.diag() /
        (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() + 1e-8)
    )
    refl_sim2    = torch.exp(torch.mm(z2, z2.t()) / tau)
    between_sim2 = torch.exp(torch.mm(z2, z1.t()) / tau)
    loss2 = -torch.log(
        between_sim2.diag() /
        (refl_sim2.sum(1) + between_sim2.sum(1) - refl_sim2.diag() + 1e-8)
    )
    return ((loss1 + loss2) * 0.5).mean()


# ── BotDCGC branch (unchanged from original) ─────────────────────────────────

def botdcgc_forward_from_xshared(model: HybridModel, X_shared, edge_index):
    Z, A_hat, A, M = model.gae(X_shared, edge_index)
    Q, P = model.clustering(Z)
    return {"Z": Z, "A_hat": A_hat, "A": A, "M": M, "Q": Q, "P": P}


def botdcgc_losses(model: HybridModel, out):
    l_recon, l_contrast = model.gae.compute_losses(
        out["Z"], out["A_hat"], out["A"], out["M"]
    )
    l_cluster = clustering_loss(out["P"], out["Q"])

    # Cluster entropy regularisation — mirrors botdcgcpipeline.py lines 90-107.
    # Prevents one cluster absorbing all nodes (clu → 0.0000 collapse).
    # Penalises imbalanced mean assignment; pushes toward H=log(K) (balanced).
    Q = out["Q"]
    Q_mean = Q.mean(dim=0).clamp(min=1e-9)
    cluster_entropy = -(Q_mean * Q_mean.log()).sum()
    max_entropy = torch.log(torch.tensor(float(Q.size(1)), device=Q.device))
    l_entropy = max_entropy - cluster_entropy   # 0 when balanced, >0 when collapsed

    loss = botdcgc_total_loss(
        l_recon, l_contrast,
        l_cluster + 0.1 * l_entropy,           # same weight as pipeline
        gamma1=model.botdcgc_cfg.gamma1,
        gamma2=model.botdcgc_cfg.gamma2,
    )
    return {
        "loss": loss,
        "reconstruction": l_recon,
        "contrastive": l_contrast,
        "clustering": l_cluster,   # log raw KL (without entropy term) for comparability
    }


@torch.no_grad()
def extract_pseudo_labels_from_xshared(
    model: HybridModel, X_shared, edge_index, conf_threshold: float
):
    model.eval()
    out          = botdcgc_forward_from_xshared(model, X_shared, edge_index)
    Q            = out["Q"]
    pseudo_labels = Q.argmax(dim=1)
    max_conf     = Q.max(dim=1).values
    pseudo_mask  = max_conf >= conf_threshold
    print(
        f"[pseudo] confident={int(pseudo_mask.sum())}/{Q.size(0)} "
        f"thr={conf_threshold} "
        + " ".join(
            [f"C{k}={int((pseudo_labels == k).sum())}"
             for k in range(model.botdcgc_cfg.n_clusters)]
        )
    )
    return pseudo_labels, pseudo_mask


# ── CACL branch — aligned with cacl_baseline_fair.py ─────────────────────────

def cacl_forward_with_fixed_xshared(
    model: HybridModel,
    graph,
    X_shared,
    pseudo_labels,
    pseudo_mask,
    train_mask,
    labels,
):
    
    # Inject real features first
    g_with_x      = model.inject_xshared(graph, X_shared)
    g_edge_dropped = _drop_edges(g_with_x, drop_rate=0.2)

    # Augment: view1 = feature dropout only, view2 = edge drop + feature dropout
    aug_g1, _ = adaptive_augment(
        g_with_x, drop_feature_rate=model.cacl_cfg.drop_feature_rate
    )
    aug_g2, _ = adaptive_augment(
        g_edge_dropped, drop_feature_rate=model.cacl_cfg.drop_feature_rate
    )

    proj1, pred1 = model.cacl_model(aug_g1)
    proj2, _     = model.cacl_model(aug_g2)

    device = pred1.device

    # ── Contrastive loss — training nodes only, minibatch to avoid OOM ────────
    l_cl = minibatch_cl_loss(
        proj1["user"][train_mask],
        proj2["user"][train_mask],
        tau=model.cacl_cfg.cl_tau,
    )

    # ── Ground truth supervised loss — all labelled training nodes ─────────────
    # Identical to CACL baseline: ensures the hybrid has at least as strong
    # a supervision signal as the standalone CACL.
    gt_labels = labels.to(device)[train_mask]
    n_total  = float(len(gt_labels))
    n_bot    = float((gt_labels == 1).sum().clamp(min=1))
    n_human  = float((gt_labels == 0).sum().clamp(min=1))
    gt_weights = torch.tensor(
        [n_total / (2 * n_human), n_total / (2 * n_bot)], device=device
    )
    l_gt = F.cross_entropy(pred1[train_mask], gt_labels, weight=gt_weights)

    # ── Pseudo-label loss — confident training nodes only ──────────────────────
    # Adds BotDCGC's structural community signal on top of ground truth.
    # Only applied where BotDCGC is confident (max Q >= PSEUDO_CONF_THR).
    conf_train_mask = pseudo_mask.to(device) & train_mask.to(device)
    l_pseudo = torch.tensor(0.0, device=device)
    if conf_train_mask.any():
        pseudo_train = pseudo_labels.to(device)[conf_train_mask]
        n_total_p = float(len(pseudo_train))
        n_pos     = float((pseudo_train == 1).sum().clamp(min=1))
        n_neg     = float((pseudo_train == 0).sum().clamp(min=1))
        pseudo_weights = torch.tensor(
            [n_total_p / (2 * n_neg), n_total_p / (2 * n_pos)], device=device
        )
        l_pseudo = F.cross_entropy(
            pred1[conf_train_mask], pseudo_train, weight=pseudo_weights
        )

    loss = (model.cacl_cfg.cl_alpha * l_cl
          + model.cacl_cfg.cl_beta  * l_gt
          + model.cacl_cfg.cl_beta  * l_pseudo)
    return {
        "loss": loss,
        "contrastive": l_cl,
        "gt_label": l_gt,
        "pseudo_label": l_pseudo,
        "pred": pred1,
    }


@torch.no_grad()
def predict_hybrid_probs(model: HybridModel, graph, X_shared):
    """Returns bot-class probabilities (float) — enables threshold tuning."""
    model.eval()
    g = model.inject_xshared(graph, X_shared)
    _, pred = model.cacl_model(g)
    return torch.softmax(pred, dim=1)[:, 1].cpu().numpy()

# ── Main training loop ────────────────────────────────────────────────────────

def train_hybrid_fixed_xshared(
    model, graph, X_shared, edge_index, labels, split, device
):
    model.to(device)
    graph      = graph.to(device)
    X_shared   = X_shared.to(device)
    edge_index = edge_index.to(device)
    labels     = labels.to(device)
    split      = split.to(device)

    train_mask = (split == 0)

    history = {
        "botdcgc_reconstruction": [],
        "botdcgc_loss": [], "botdcgc_contrastive": [], "botdcgc_clustering": [],
        "cacl_loss": [], "cacl_contrastive": [], "cacl_gt_label": [], "cacl_pseudo_label": [],
        "logged_epochs": [],
        "train_accuracy": [], "train_balanced_accuracy": [], "train_f1": [],
        "train_macro_f1": [], "train_ari": [], "train_nmi": [],
        "val_accuracy":   [], "val_balanced_accuracy":   [], "val_f1": [],
        "val_macro_f1":   [], "val_ari": [],              "val_nmi": [],
        "test_accuracy":  [], "test_balanced_accuracy":  [], "test_f1": [],
        "test_macro_f1":  [], "test_ari": [],             "test_nmi": [],
    }

    print("=" * 60)
    print("Hybrid fair run — fixed X_shared")
    print(f"CL batch size : {CL_BATCH_SIZE}  |  CACL patience: {PATIENCE} evals")
    print("=" * 60)
    opt_bot = optim.Adam(
    list(model.gae.parameters()) + list(model.clustering.parameters()),
    lr=BOTDCGC_LR, weight_decay=BOTDCGC_WD,
    )

    # ── Phase 1: BotDCGC reconstruction pretrain ──────────────────────────────
    print(f"[Phase 1] BotDCGC reconstruction pretrain: {PRETRAIN_EPOCHS} epochs")
    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.gae.train()
        opt_bot.zero_grad()
        out    = botdcgc_forward_from_xshared(model, X_shared, edge_index)
        l_recon = reconstruction_loss(out["A"], out["A_hat"])
        l_recon.backward()
        opt_bot.step()
        history["botdcgc_reconstruction"].append(l_recon.item())
        if ep % 10 == 0 or ep == 1:
            print(f"  pretrain [{ep:03d}/{PRETRAIN_EPOCHS}] rec={l_recon.item():.4f}")

    # ── Phase 2: BotDCGC joint training ──────────────────────────────────────
    print("[Phase 2] Initialising BotDCGC clusters with K-means")
    with torch.no_grad():
        out = botdcgc_forward_from_xshared(model, X_shared, edge_index)
        model.clustering.initialise(out["Z"])
    print(f"[Phase 2] BotDCGC joint training: {BOTDCGC_EPOCHS} epochs")
    for ep in range(1, BOTDCGC_EPOCHS + 1):
        model.gae.train(); model.clustering.train()
        opt_bot.zero_grad()
        out    = botdcgc_forward_from_xshared(model, X_shared, edge_index)
        losses = botdcgc_losses(model, out)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(                         
            list(model.gae.parameters()) +                       
            list(model.clustering.parameters()), max_norm=1.0  
        )     
        opt_bot.step()
        # Without this, centres drift as the GAT updates and the KL loss  
        # sees stale targets, causing clu → 0 collapse.                   
        if ep % 10 == 0:                                        
            with torch.no_grad():                             
                out_refresh = botdcgc_forward_from_xshared(   
                    model, X_shared, edge_index                 
                )                                              
                model.clustering.initialise(out_refresh["Z"])   
        history["botdcgc_loss"].append(losses["loss"].item())
        history["botdcgc_contrastive"].append(losses["contrastive"].item())
        history["botdcgc_clustering"].append(losses["clustering"].item())
        if ep % LOG_EVERY == 0 or ep == 1:
            print(
                f"  botdcgc [{ep:03d}/{BOTDCGC_EPOCHS}] "
                f"loss={losses['loss'].item():.4f}  "
                f"rec={losses['reconstruction'].item():.4f}  "
                f"con={losses['contrastive'].item():.4f}  "
                f"clu={losses['clustering'].item():.4f}"
            )

    pseudo_labels, pseudo_mask = extract_pseudo_labels_from_xshared(
        model, X_shared, edge_index, conf_threshold=PSEUDO_CONF_THR
    )

    # ── Phase 3: CACL pseudo-label training with early stopping ──────────────
    opt_cacl = optim.Adam(
        model.cacl_model.parameters(), lr=CACL_LR, weight_decay=CACL_WD
    )

    best_val_f1       = -1.0
    best_epoch        = 0
    best_state        = None
    epochs_no_improve = 0

    labels_np = labels.detach().cpu().numpy()
    split_np  = split.detach().cpu().numpy()
    val_mask_np = split_np == 1

    thresholds = np.arange(0.05, 0.70, 0.01)

    print(f"[Phase 3] CACL pseudo-label training: {CACL_EPOCHS} epochs  "
          f"patience={PATIENCE}")
    for ep in range(1, CACL_EPOCHS + 1):
        model.cacl_model.train()
        opt_cacl.zero_grad()
        out = cacl_forward_with_fixed_xshared(
            model, graph, X_shared, pseudo_labels, pseudo_mask, train_mask, labels
        )
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.cacl_model.parameters(), max_norm=1.0)
        opt_cacl.step()

        history["cacl_loss"].append(out["loss"].item())
        history["cacl_contrastive"].append(out["contrastive"].item())
        history["cacl_gt_label"].append(out["gt_label"].item())
        history["cacl_pseudo_label"].append(out["pseudo_label"].item())

        if ep % LOG_EVERY == 0 or ep == 1:
            raw_probs = predict_hybrid_probs(model, graph, X_shared)

            # Val-optimal threshold for early stopping
            best_t = max(
                thresholds,
                key=lambda t: f1_score(
                    labels_np[val_mask_np],
                    raw_probs[val_mask_np] > t,
                    zero_division=0,
                ),
            )
            binary_pred = (raw_probs > best_t).astype(int)

            metrics, mapping = evaluate_all_splits_common(
                y_true_all=labels_np,
                y_pred_raw_all=binary_pred,
                split_all=split_np,
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
                f"  cacl [{ep:03d}/{CACL_EPOCHS}] "
                f"loss={out['loss'].item():.4f}  "
                f"CL={out['contrastive'].item():.4f}  "
                f"pseudo={out['pseudo_label'].item():.4f}  "
                f"thr={best_t:.2f}  "
                f"train_f1={metrics['train']['f1']:.4f}  "
                f"val_f1={metrics['val']['f1']:.4f}  "
                f"test_f1={metrics['test']['f1']:.4f}"
                f"{marker}"
            )

            if epochs_no_improve >= PATIENCE:
                print(
                    f"\n[early stop] epoch {ep}: no val_f1 improvement "
                    f"for {PATIENCE} evals. Best was epoch {best_epoch} "
                    f"val_f1={best_val_f1:.4f}"
                )
                break

        if ep % 10 == 0:
            gc.collect()

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"\n[train] Restored best model from epoch {best_epoch} "
              f"(val_f1={best_val_f1:.4f})")

    # Final evaluation with val-tuned threshold
    raw_probs = predict_hybrid_probs(model, graph, X_shared)
    best_t = max(
        thresholds,
        key=lambda t: f1_score(
            labels_np[val_mask_np],
            raw_probs[val_mask_np] > t,
            zero_division=0,
        ),
    )
    print(f"[threshold] Best val threshold: {best_t:.2f}")
    binary_pred = (raw_probs > best_t).astype(int)

    final, mapping = evaluate_all_splits_common(
        y_true_all=labels_np,
        y_pred_raw_all=binary_pred,
        split_all=split_np,
        use_train_mapping=False,
        n_clusters=2,
    )

    return {
        "history": history,
        "final": final,
        "raw_pred": binary_pred,
        "mapping": mapping,
        "threshold": float(best_t),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start    = time.time()
    raw      = torch.load(DATA_PATH, weights_only=False)
    X_shared = torch.load(X_SHARED_PATH, weights_only=False).float()
    load_x_time = time.time() - start

    labels     = raw["labels"]
    split      = raw["split"]
    edge_index = raw["edge_index"]
    user_ids   = raw["user_ids"]

    graph    = build_user_graph(len(user_ids), edge_index, x_dim=X_shared.shape[1])
    metadata = (graph.node_types, graph.edge_types)

    feat_cfg = FeatureEncoderCfg(embed_dim=X_shared.shape[1])
    bot_cfg  = BotDCGCCfg(
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
    cacl_cfg = CACLCfg(
        hetero_model="SAGE",
        hidden_channels=64,
        out_channels=16,
        proj_out=16,
        proj_hidden=32,
        num_layer=1,
    )
    model = HybridModel(
        metadata=metadata,
        in_channels={"user": X_shared.shape[1]},
        feat_cfg=feat_cfg,
        botdcgc_cfg=bot_cfg,
        cacl_cfg=cacl_cfg,
    )

    train_start = time.time()
    result      = train_hybrid_fixed_xshared(
        model, graph, X_shared, edge_index, labels, split, device
    )
    train_time  = time.time() - train_start

    model_path = os.path.join(RUN_DIR, "hybrid_model.pt")
    torch.save(
        {
            "gae":          model.gae.state_dict(),
            "clustering":   model.clustering.state_dict(),
            "cacl_model":   model.cacl_model.state_dict(),
            "model_config": {
                "x_shared_dim":     int(X_shared.shape[1]),
                "gat_hidden_dims":  bot_cfg.gat_hidden_dims,
                "n_clusters":       bot_cfg.n_clusters,
                "hetero_model":     cacl_cfg.hetero_model,
                "hidden_channels":  cacl_cfg.hidden_channels,
                "out_channels":     cacl_cfg.out_channels,
                "proj_out":         cacl_cfg.proj_out,
                "proj_hidden":      cacl_cfg.proj_hidden,
                "num_layer":        cacl_cfg.num_layer,
            },
            "final_metrics":   result["final"],
            "history":         result["history"],
            "cluster_mapping": result["mapping"],
            "threshold":       result["threshold"],
        },
        model_path,
    )

    save_common_outputs(
        run_dir=RUN_DIR,
        model_name="hybrid_fixed_xshared_notemp",
        user_ids=user_ids,
        labels=labels,
        split=split,
        raw_predictions=result["raw_pred"],
        final_metrics=result["final"],
        mapping=result["mapping"],
        history=result["history"],
        timing={
            "x_shared_load_seconds":              load_x_time,
            "training_and_evaluation_seconds":    train_time,
            "total_experiment_seconds":           load_x_time + train_time,
        },
        model=model,
        x_shared_path=X_SHARED_PATH,
    )
    print(f"Saved fair Hybrid run to: {RUN_DIR}")


if __name__ == "__main__":
    main()