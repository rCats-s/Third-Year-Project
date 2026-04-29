
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.data import HeteroData

# ── local imports ────────────────────────────────────────────────────────────
from botdcgcpipeline import cluster_accuracy
from clustering import ClusteringModule, clustering_loss, total_loss as botdcgc_total_loss
from CL_model import (
    Classifier,
    HeteroGraphConvModel,
    HGcnCLModel,
    MLPProjector,
    adaptive_augment,
    compute_loss as cacl_compute_loss,
)
from feature_encoder import UserFeatureEncoder
from graph_autoencoder import (
    BotDCGCGraphAutoEncoder,
    reconstruction_loss,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Hyper-parameter containers (avoids long __init__ signatures)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureEncoderCfg:
    num_features:    int   = 7   # 7 static + 6 temporal numerical
    cat_features:    int   = 11   # 11 static + 4 temporal categorical
    roberta_dim:     int   = 768
    lstm_hidden:     int   = 128
    embed_dim:       int   = 128        # D  (must be divisible by 4)


@dataclass
class BotDCGCCfg:
    gat_hidden_dims: list  = field(default_factory=lambda: [128, 128, 16])
    n_clusters:      int   = 2
    order_t:         int   = 10
    alpha:           float = 0.2        # LeakyReLU slope in GAT
    tau:             float = 1.0        # contrastive temperature
    gamma1:          float = 2.0        # weight of L_contrastive
    gamma2:          float = 10.0       # weight of L_clustering
    undirected:      bool  = True
    add_self_loops:  bool  = False


@dataclass
class CACLCfg:
    hetero_model:    str   = "SAGE"      # HGT | GAT | SAGE | RGT
    hidden_channels: int   = 128
    out_channels:    int   = 32
    num_layer:       int   = 2
    momentum:        float = 0.1
    dropout:         float = 0.5
    proj_hidden:     int   = 128
    proj_out:        int   = 32
    cl_tau:          float = 0.5        # CACL contrastive temperature
    cl_alpha:        float = 0.01       # weight of L_contrastive in CACL
    cl_beta:         float = 1.0        # weight of pseudo-label loss in CACL
    drop_feature_rate: float = 0.2


@dataclass
class TrainCfg:
    # Phase C – BotDCGC pre-train
    pretrain_epochs:   int   = 30
    pretrain_lr:       float = 5e-3
    pretrain_wd:       float = 5e-3
    botdcgc_epochs:    int   = 100
    botdcgc_lr:        float = 5e-3
    botdcgc_wd:        float = 5e-3
    # Phase F – CACL fine-tuning
    cacl_epochs:       int   = 100
    cacl_lr:           float = 1e-3
    cacl_wd:           float = 1e-4
    pseudo_conf_thr:   float = 0.80
    log_every:         int   = 10




class _Args:
    """Minimal namespace that CL_model.HeteroGraphConvModel.__init__ expects."""
    def __init__(self, cfg: CACLCfg):
        self.num_layer = cfg.num_layer
        self.momentum  = cfg.momentum
        self.dropout   = cfg.dropout



class HybridModel(nn.Module):
  

    def __init__(
        self,
        # graph metadata needed to build HeteroGraphConvModel
        metadata: Tuple,                # (node_types, edge_types)
        in_channels: Dict[str, int],    # per-node-type raw feature dims
        #
        feat_cfg:    FeatureEncoderCfg = None,
        botdcgc_cfg: BotDCGCCfg       = None,
        cacl_cfg:    CACLCfg           = None,
    ):
        super().__init__()

        feat_cfg    = feat_cfg    or FeatureEncoderCfg()
        botdcgc_cfg = botdcgc_cfg or BotDCGCCfg()
        cacl_cfg    = cacl_cfg    or CACLCfg()

        self.feat_cfg    = feat_cfg
        self.botdcgc_cfg = botdcgc_cfg
        self.cacl_cfg    = cacl_cfg

        # ── 4.1  Shared user feature encoder ─────────────────────────────
        self.user_encoder = UserFeatureEncoder(
            num_features = feat_cfg.num_features,
            cat_features = feat_cfg.cat_features,
            roberta_dim  = feat_cfg.roberta_dim,
            lstm_hidden  = feat_cfg.lstm_hidden,
            embed_dim    = feat_cfg.embed_dim,
        )

      
        embed_dim = feat_cfg.embed_dim
        self.gae = BotDCGCGraphAutoEncoder(
            in_dim          = embed_dim,
            gat_hidden_dims = botdcgc_cfg.gat_hidden_dims,
            order_t         = botdcgc_cfg.order_t,
            alpha           = botdcgc_cfg.alpha,
            tau             = botdcgc_cfg.tau,
            undirected      = botdcgc_cfg.undirected,
            add_self_loops  = botdcgc_cfg.add_self_loops,
        )

        
        gat_last_dim = botdcgc_cfg.gat_hidden_dims[-1]
        self.clustering = ClusteringModule(
            n_clusters = botdcgc_cfg.n_clusters,
            embed_dim  = gat_last_dim,
        )

      
        cacl_in_channels = dict(in_channels)
        cacl_in_channels['user'] = embed_dim

        _args = _Args(cacl_cfg)
        hetero_encoder = HeteroGraphConvModel(
            model           = cacl_cfg.hetero_model,
            in_channels     = cacl_in_channels,
            hidden_channels = cacl_cfg.hidden_channels,
            out_channels    = cacl_cfg.out_channels,
            metadata        = metadata,
            args            = _args,
        )

       
        hetero_out_dim = cacl_cfg.out_channels + embed_dim

        projector = MLPProjector(
            node_types  = [n for n in metadata[0]],
            input_size  = {n: hetero_out_dim for n in metadata[0]},
            output_size = cacl_cfg.proj_out,
            hidden_size = cacl_cfg.proj_hidden,
        )

        classifier = Classifier(
            input_dim  = hetero_out_dim,
            hidden_dim = 64,
        )

        self.cacl_model = HGcnCLModel(
            encoder    = hetero_encoder,
            projector  = projector,
            classifier = classifier,
        )

        # ── Internal state set during training ───────────────────────────
        self._pseudo_labels: Optional[torch.Tensor] = None   # (N,) long
        self._pseudo_mask:   Optional[torch.Tensor] = None   # (N,) bool




    def botdcgc_forward(
        self,
        num:       torch.Tensor,
        cat:       torch.Tensor,
        desc_emb:  torch.Tensor,
        tweet_emb: torch.Tensor,
        tweet_len: torch.Tensor,
        edge_index: torch.Tensor,        # (2, E)  user-user edges
    ) -> Dict[str, torch.Tensor]:
        """
        Full BotDCGC forward pass.
        Returns a dict with keys: X, Z, A_hat, A, M, Q, P.
        """
        X = self.user_encoder(num, cat, desc_emb, tweet_emb, tweet_len)
        Z, A_hat, A, M = self.gae(X, edge_index)
        Q, P = self.clustering(Z)
        return dict(X=X, Z=Z, A_hat=A_hat, A=A, M=M, Q=Q, P=P)

    def botdcgc_losses(
        self,
        out: Dict[str, torch.Tensor],
        recon_pos_weight: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        l_recon, l_contrast = self.gae.compute_losses(
            out["Z"], out["A_hat"], out["A"], out["M"],
            recon_pos_weight=recon_pos_weight,
        )
        l_cluster = clustering_loss(out["P"], out["Q"])
        l_total   = botdcgc_total_loss(
            l_recon, l_contrast, l_cluster,
            gamma1=self.botdcgc_cfg.gamma1,
            gamma2=self.botdcgc_cfg.gamma2,
        )
        return dict(loss=l_total, reconstruction=l_recon,
                    contrastive=l_contrast, clustering=l_cluster)

    @torch.no_grad()
    def extract_pseudo_labels(
        self,
        num:       torch.Tensor,
        cat:       torch.Tensor,
        desc_emb:  torch.Tensor,
        tweet_emb: torch.Tensor,
        tweet_len: torch.Tensor,
        edge_index: torch.Tensor,
        conf_threshold: float = 0.80,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        self.eval()
        out = self.botdcgc_forward(num, cat, desc_emb, tweet_emb, tweet_len, edge_index)
        Q = out["Q"]                              # (N, K) soft assignments
        pseudo_labels = Q.argmax(dim=1)           # (N,)
        max_conf, _   = Q.max(dim=1)              # (N,)
        pseudo_mask   = max_conf >= conf_threshold

        self._pseudo_labels = pseudo_labels.cpu()
        self._pseudo_mask   = pseudo_mask.cpu()

        n_confident = pseudo_mask.sum().item()
        print(
            f"[extract_pseudo_labels] {n_confident}/{Q.size(0)} nodes confident "
            f"(thr={conf_threshold}).  "
            f"Cluster distribution: "
            + ", ".join(
                f"C{k}={int((pseudo_labels == k).sum())}"
                for k in range(self.botdcgc_cfg.n_clusters)
            )
        )
        return pseudo_labels, pseudo_mask


    @staticmethod
    def inject_xshared(graph: HeteroData, X_shared: torch.Tensor) -> HeteroData:
        """
        Replace graph['user'].x with X_shared so the CACL encoder
        sees the same shared representation that BotDCGC produced.

        A shallow copy is made – the underlying edge tensors are shared.
        """
        g = copy.copy(graph)
        g['user'].x = X_shared
        # keep existing edge types / other attrs untouched
        return g

   
    def cacl_forward_with_pseudo(
        self,
        graph:         HeteroData,
        num:           torch.Tensor,
        cat:           torch.Tensor,
        desc_emb:      torch.Tensor,
        tweet_emb:     torch.Tensor,
        tweet_len:     torch.Tensor,
        pseudo_labels: torch.Tensor,
        pseudo_mask:   torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        device = next(self.cacl_model.parameters()).device

        # Phase D: frozen shared encoder
        with torch.no_grad():
            X_shared = self.user_encoder(num, cat, desc_emb, tweet_emb, tweet_len)

        # Augment original graph first
        aug_g1, aug_g2 = adaptive_augment(
            graph,
            drop_feature_rate=self.cacl_cfg.drop_feature_rate
        )

        # Inject X_shared into both augmented views
        aug_g1 = self.inject_xshared(aug_g1, X_shared)
        aug_g2 = self.inject_xshared(aug_g2, X_shared)

        # CACL forward on both views
        proj1, pred1 = self.cacl_model(aug_g1)
        proj2, _     = self.cacl_model(aug_g2)

        user_emb1 = proj1["user"]     # (N, proj_out)
        user_emb2 = proj2["user"]     # (N, proj_out)
        user_pred = pred1             # (N, 2)

        # Contrastive loss
        N = user_emb1.size(0)
        split = torch.zeros(N, dtype=torch.long, device=device)

        l_cl = cacl_compute_loss(
            emb1=user_emb1,
            emb2=user_emb2,
            pred=None,   # only contrastive term here
            label=pseudo_labels[:N].to(device),
            split=split,
            tau=self.cacl_cfg.cl_tau,
        )

        # Pseudo-label classification loss on confident nodes only
        conf_mask = pseudo_mask[:N].to(device)
        l_pseudo = torch.tensor(0.0, device=device)
        if conf_mask.any() and user_pred is not None:
            preds_conf = user_pred[conf_mask]
            labels_conf = pseudo_labels[:N][conf_mask].to(device)
            l_pseudo = F.cross_entropy(preds_conf, labels_conf)

        # Total Phase F loss
        l_total = self.cacl_cfg.cl_alpha * l_cl + self.cacl_cfg.cl_beta * l_pseudo

        return {
            "loss":         l_total,
            "contrastive":  l_cl,
            "pseudo_label": l_pseudo,
            "user_emb1":    user_emb1,
            "user_pred":    user_pred,
        }


    @torch.no_grad()
    def predict(
        self,
        graph:     HeteroData,
        num:       torch.Tensor,
        cat:       torch.Tensor,
        desc_emb:  torch.Tensor,
        tweet_emb: torch.Tensor,
        tweet_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns hard bot/human predictions (0 or 1) via the CACL classifier.
        """
        self.eval()
        X_shared = self.user_encoder(num, cat, desc_emb, tweet_emb, tweet_len)
        g = self.inject_xshared(graph, X_shared)
        _, pred = self.cacl_model(g)
        return pred.argmax(dim=1)    # (N,)



#  freeze / unfreeze helpers


def _set_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def freeze_botdcgc(model: HybridModel) -> None:
    """Freeze user_encoder, gae, clustering for Phase F."""
    _set_grad(model.user_encoder, False)
    _set_grad(model.gae,          False)
    _set_grad(model.clustering,   False)


def unfreeze_all(model: HybridModel) -> None:
    _set_grad(model, True)



#  Full training routine


def train_hybrid(
    model:      HybridModel,
    # flat user-feature tensors (all nodes)
    num:        torch.Tensor,
    cat:        torch.Tensor,
    desc_emb:   torch.Tensor,
    tweet_emb:  torch.Tensor,
    tweet_len:  torch.Tensor,
    # user-level edge index (for BotDCGC)
    user_edge_index: torch.Tensor,
    # full HeteroData graph (for CACL)
    graph:      HeteroData,
    # optional ground-truth for logging only
    labels:     Optional[torch.Tensor] = None,
    device:     str  = "cpu",
    cfg:        TrainCfg = None,
) -> Dict:
    
    cfg = cfg or TrainCfg()
    model.to(device)

    # move tensors to device
    num            = num.to(device)
    cat            = cat.to(device)
    desc_emb       = desc_emb.to(device)
    tweet_emb      = tweet_emb.to(device)
    tweet_len      = tweet_len.to(device)
    user_edge_index = user_edge_index.to(device)

    history = {
        "phase_c_recon":   [],
        "phase_c_loss":    [],
        "phase_f_cl":      [],
        "phase_f_pseudo":  [],
        "phase_f_total":   [],
        "val_accuracy":    [],
        "val_f1":          [],
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase C-1: Reconstruction-only pre-train (stabilises encoder)
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Phase C-1  BotDCGC reconstruction pre-training")
    print("=" * 60)

    pretrain_opt = optim.Adam(
        list(model.user_encoder.parameters()) + list(model.gae.parameters()),
        lr=cfg.pretrain_lr, weight_decay=cfg.pretrain_wd,
    )

    for ep in range(1, cfg.pretrain_epochs + 1):
        model.train()
        pretrain_opt.zero_grad()

        X = model.user_encoder(num, cat, desc_emb, tweet_emb, tweet_len)
        Z, A_hat, A, M = model.gae(X, user_edge_index)
        l = reconstruction_loss(A, A_hat)
        l.backward()
        pretrain_opt.step()

        history["phase_c_recon"].append(l.item())
        if ep % cfg.log_every == 0 or ep == 1:
            print(f"  [C-1 {ep:03d}/{cfg.pretrain_epochs}]  L_recon={l.item():.4f}")


    print("\nPhase C-2  K-means cluster initialisation")
    model.eval()
    with torch.no_grad():
        X = model.user_encoder(num, cat, desc_emb, tweet_emb, tweet_len)
        Z, _, _, _ = model.gae(X, user_edge_index)
    model.clustering.initialise(Z)

    print("\nPhase C-3  BotDCGC joint training")
    print("=" * 60)

    joint_opt = optim.Adam(
        list(model.user_encoder.parameters())
        + list(model.gae.parameters())
        + list(model.clustering.parameters()),
        lr=cfg.botdcgc_lr, weight_decay=cfg.botdcgc_wd,
    )

    for ep in range(1, cfg.botdcgc_epochs + 1):
        model.train()
        joint_opt.zero_grad()

        out    = model.botdcgc_forward(num, cat, desc_emb, tweet_emb, tweet_len, user_edge_index)
        losses = model.botdcgc_losses(out)
        losses["loss"].backward()
        joint_opt.step()

        history["phase_c_loss"].append(losses["loss"].item())
        if ep % cfg.log_every == 0 or ep == 1:
            print(
                f"  [C-3 {ep:03d}/{cfg.botdcgc_epochs}]"
                f"  total={losses['loss'].item():.4f}"
                f"  recon={losses['reconstruction'].item():.4f}"
                f"  contrast={losses['contrastive'].item():.4f}"
                f"  cluster={losses['clustering'].item():.4f}"
            )

    print("\nPhase C-4  Pseudo-label extraction")
    pseudo_labels, pseudo_mask = model.extract_pseudo_labels(
        num, cat, desc_emb, tweet_emb, tweet_len, user_edge_index,
        conf_threshold=cfg.pseudo_conf_thr,
    )
    pseudo_labels = pseudo_labels.to(device)
    pseudo_mask   = pseudo_mask.to(device)
    
    print("\nPhase F  CACL contrastive + pseudo-label training")
    print("=" * 60)

    freeze_botdcgc(model)

    cacl_opt = optim.Adam(
        model.cacl_model.parameters(),
        lr=cfg.cacl_lr, weight_decay=cfg.cacl_wd,
    )

    for ep in range(1, cfg.cacl_epochs + 1):
        model.train()
        cacl_opt.zero_grad()

        out = model.cacl_forward_with_pseudo(
            graph, num, cat, desc_emb, tweet_emb, tweet_len,
            pseudo_labels, pseudo_mask,
        )
        out["loss"].backward()
        cacl_opt.step()

        history["phase_f_total"].append(out["loss"].item())
        history["phase_f_cl"].append(out["contrastive"].item())
        history["phase_f_pseudo"].append(out["pseudo_label"].item())

        if (ep % cfg.log_every == 0 or ep == 1) and labels is not None:
            metrics = _eval(model, graph, num, cat, desc_emb, tweet_emb,
                            tweet_len, labels, device, pseudo_labels)
            history["val_accuracy"].append(metrics["accuracy"])
            history["val_f1"].append(metrics["f1"])
            print(
                f"  [F {ep:03d}/{cfg.cacl_epochs}]"
                f"  total={out['loss'].item():.4f}"
                f"  CL={out['contrastive'].item():.4f}"
                f"  pseudo={out['pseudo_label'].item():.4f}"
                f"  acc={metrics['accuracy']:.4f}"
                f"  f1={metrics['f1']:.4f}"
            )
        elif ep % cfg.log_every == 0 or ep == 1:
            print(
                f"  [F {ep:03d}/{cfg.cacl_epochs}]"
                f"  total={out['loss'].item():.4f}"
                f"  CL={out['contrastive'].item():.4f}"
                f"  pseudo={out['pseudo_label'].item():.4f}"
            )

    # Restore gradients for all params (e.g. for future fine-tuning)
    unfreeze_all(model)

    final = {}
    if labels is not None:
        final = _eval(model, graph, num, cat, desc_emb, tweet_emb,
                      tweet_len, labels, device, pseudo_labels)
        print("\nFinal evaluation:")
        for k, v in final.items():
            print(f"  {k:12s}: {v:.4f}")

    return {"history": history, "final": final}


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation helper
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval(
    model,
    graph,
    num, cat, desc_emb, tweet_emb, tweet_len,
    labels:        torch.Tensor,
    device:        str,
    pseudo_labels: torch.Tensor,
) -> Dict:
    model.eval()
    preds = model.predict(graph, num, cat, desc_emb, tweet_emb, tweet_len)

    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    # Hungarian matching between cluster ids and true labels
    n_clusters = model.botdcgc_cfg.n_clusters
    _, y_remap = cluster_accuracy(y_true, y_pred, n_clusters=n_clusters)

    return {
        "accuracy":  accuracy_score(y_true, y_remap),
        "precision": precision_score(y_true, y_remap, average="binary", zero_division=0),
        "recall":    recall_score(y_true, y_remap, average="binary", zero_division=0),
        "f1":        f1_score(y_true, y_remap, average="binary", zero_division=0),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Quick smoke-test  (python hybrid_model.py)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    from torch_geometric.data import HeteroData

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = torch.load(
        r"D:\3rd year project\Implementation\Datasets\twibot22_smoke\graph_data_model_ready_real.pt"
    )

    num        = raw["num"]
    cat        = raw["cat"]
    desc_emb   = raw["desc_emb"]
    tweet_emb  = raw["tweet_emb"]
    tweet_len  = raw["tweet_len"]
    labels     = raw["labels"]
    user_ei    = raw["edge_index"]

    print("\n===== HYBRID SANITY CHECK =====")
    print("num shape      :", num.shape)
    print("cat shape      :", cat.shape)
    print("desc_emb shape :", desc_emb.shape)
    print("tweet_emb shape:", tweet_emb.shape)
    print("tweet_len shape:", tweet_len.shape)
    print("labels shape   :", labels.shape)
    print("edge_index shape:", user_ei.shape)

    assert num.shape[1] == 13, f"Expected 13 numerical features, got {num.shape[1]}"
    assert cat.shape[1] == 15, f"Expected 15 categorical features, got {cat.shape[1]}"

    print("num has nan:", torch.isnan(num).any().item())
    print("cat has nan:", torch.isnan(cat).any().item())
    print("desc_emb has nan:", torch.isnan(desc_emb).any().item())
    print("tweet_emb has nan:", torch.isnan(tweet_emb).any().item())

    graph = HeteroData()
    graph['user'].x = torch.zeros(num.size(0), 256)   # placeholder, overwritten by X_shared
    graph[('user', 'follows', 'user')].edge_index = user_ei

    metadata = (graph.node_types, graph.edge_types)
    in_channels = {'user': 256}

    feat_cfg = FeatureEncoderCfg(
        num_features=13,
        cat_features=15,
        roberta_dim=768,
        lstm_hidden=128,
        embed_dim=256,
    )

    bd_cfg = BotDCGCCfg(
        gat_hidden_dims=[256, 256, 16],
        n_clusters=2,
        order_t=10,
        alpha=0.2,
        tau=1.0,
        gamma1=2.0,
        gamma2=10.0,
        undirected=True,
        add_self_loops=False,
    )

    cl_cfg = CACLCfg(
        hetero_model="SAGE",
        hidden_channels=64,
        out_channels=16,
        proj_out=16,
        proj_hidden=32,
        num_layer=1,
    )

    model = HybridModel(metadata, in_channels, feat_cfg, bd_cfg, cl_cfg)

    train_cfg = TrainCfg(
        pretrain_epochs=1,
        botdcgc_epochs=1,
        cacl_epochs=1,
        log_every=1,
        pseudo_conf_thr=0.80,
    )

    result = train_hybrid(
        model,
        num, cat, desc_emb, tweet_emb, tweet_len,
        user_ei, graph,
        labels=labels,
        device=device,
        cfg=train_cfg,
    )

    print("\n[smoke-test] history keys:", list(result["history"].keys()))
    print("[smoke-test] final keys:", list(result["final"].keys()))
    print("[smoke-test] PASSED ✓")