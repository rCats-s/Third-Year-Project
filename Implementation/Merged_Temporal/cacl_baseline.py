

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import HeteroData

# ── local imports ─────────────────────────────────────────────────────────────
from CL_model import (
    Classifier,
    HeteroGraphConvModel,
    HGcnCLModel,
    MLPProjector,
    adaptive_augment,
)
from CL_utils import unsupervised_cl_loss
from feature_encoder import UserFeatureEncoder




@dataclass
class FeatureEncoderCfg:
    """Shared feature-encoder hyper-parameters."""
    num_features: int   = 13   # 7 static + 6 temporal numerical
    cat_features: int   = 15   # 11 static + 4 temporal categorical
    roberta_dim:  int   = 768
    lstm_hidden:  int   = 128
    embed_dim:    int   = 256       # D; must be divisible by 4


@dataclass
class CACLCfg:
    """
    CACL encoder/projector/classifier hyper-parameters.

    Note on cl_beta
    ---------------
    CACLBaseline  →  weights cross-entropy against *ground-truth* labels.
    HybridModel   →  weights cross-entropy against *pseudo* labels.
    The field name is intentionally the same so a single CACLCfg instance
    can be passed to both models; only its semantic role differs.
    """
    hetero_model:     str   = "SAGE"     # HGT | GAT | SAGE | RGT
    hidden_channels:  int   = 256
    out_channels:     int   = 64
    num_layer:        int   = 2
    momentum:         float = 0.1
    dropout:          float = 0.5
    proj_hidden:      int   = 256
    proj_out:         int   = 64
    cl_tau:           float = 0.5      # contrastive temperature
    cl_alpha:         float = 0.01     # weight of L_contrastive
    cl_beta:          float = 1.0      # weight of L_classification (true labels)
    drop_feature_rate: float = 0.2


@dataclass
class BaselineTrainCfg:
    """Training knobs for the CACL baseline (single-phase, no BotDCGC)."""
    epochs:   int   = 200
    lr:       float = 1e-3
    wd:       float = 1e-4
    log_every: int  = 10




class _Args:
    def __init__(self, cfg: CACLCfg):
        self.num_layer = cfg.num_layer
        self.momentum  = cfg.momentum
        self.dropout   = cfg.dropout


# ═══════════════════════════════════════════════════════════════════════════
#  CACLBaseline
# ═══════════════════════════════════════════════════════════════════════════

class CACLBaseline(nn.Module):
   

    def __init__(
        self,
        metadata:        Tuple,
        in_channels:     Dict[str, int],
        feat_cfg:        FeatureEncoderCfg = None,
        cacl_cfg:        CACLCfg           = None,
        use_classifier:  bool              = True,
    ):
        super().__init__()

        self.feat_cfg = feat_cfg or FeatureEncoderCfg()
        self.cacl_cfg = cacl_cfg or CACLCfg()
        self.use_classifier = use_classifier

        embed_dim = self.feat_cfg.embed_dim

        # ── Section 4.1: shared feature encoder ──────────────────────────
        self.user_encoder = UserFeatureEncoder(
            num_features = self.feat_cfg.num_features,
            cat_features = self.feat_cfg.cat_features,
            roberta_dim  = self.feat_cfg.roberta_dim,
            lstm_hidden  = self.feat_cfg.lstm_hidden,
            embed_dim    = embed_dim,
        )

        # ── CACL encoder (hetero GNN) ─────────────────────────────────────
        # Override 'user' input dim: the GNN always sees X_shared, not raw features
        cacl_in_channels = dict(in_channels)
        cacl_in_channels['user'] = embed_dim

        _args = _Args(self.cacl_cfg)
        hetero_encoder = HeteroGraphConvModel(
            model           = self.cacl_cfg.hetero_model,
            in_channels     = cacl_in_channels,
            hidden_channels = self.cacl_cfg.hidden_channels,
            out_channels    = self.cacl_cfg.out_channels,
            metadata        = metadata,
            args            = _args,
        )

        # HeteroGraphConvModel concatenates conv output with raw input features,
        # so the effective output dim per node type is out_channels + embed_dim.
        hetero_out_dim = self.cacl_cfg.out_channels + embed_dim

        projector = MLPProjector(
            node_types  = list(metadata[0]),
            input_size  = {n: hetero_out_dim for n in metadata[0]},
            output_size = self.cacl_cfg.proj_out,
            hidden_size = self.cacl_cfg.proj_hidden,
        )

        classifier = (
            Classifier(input_dim=hetero_out_dim, hidden_dim=64)
            if use_classifier
            else None
        )

        self.cacl_model = HGcnCLModel(
            encoder    = hetero_encoder,
            projector  = projector,
            classifier = classifier,
        )

    

    @staticmethod
    def inject_xshared(graph: HeteroData, X_shared: torch.Tensor) -> HeteroData:
        """
        Shallow-copy graph and replace graph['user'].x with X_shared.
        All edge tensors remain shared.
        """
        g = copy.copy(graph)
        g['user'].x = X_shared
        return g

   

    def forward(
        self,
        graph:     HeteroData,
        num:       torch.Tensor,
        cat:       torch.Tensor,
        desc_emb:  torch.Tensor,
        tweet_emb: torch.Tensor,
        tweet_len: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
       
        X_shared = self.user_encoder(num, cat, desc_emb, tweet_emb, tweet_len)
        g = self.inject_xshared(graph, X_shared)
        proj_dict, pred = self.cacl_model(g)
        return {
            "X_shared": X_shared,
            "proj":     proj_dict['user'],
            "pred":     pred,
        }

  
    def forward_two_views(
    self,
    graph:     HeteroData,
    num:       torch.Tensor,
    cat:       torch.Tensor,
    desc_emb:  torch.Tensor,
    tweet_emb: torch.Tensor,
    tweet_len: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        X_shared = self.user_encoder(num, cat, desc_emb, tweet_emb, tweet_len)

        # Augment the original graph first
        aug_g1, aug_g2 = adaptive_augment(
            graph,
            drop_feature_rate=self.cacl_cfg.drop_feature_rate
        )

        # Then inject shared user embeddings into each augmented view
        aug_g1 = self.inject_xshared(aug_g1, X_shared)
        aug_g2 = self.inject_xshared(aug_g2, X_shared)

        proj1, pred = self.cacl_model(aug_g1)
        proj2, _    = self.cacl_model(aug_g2)

        return {
            "X_shared": X_shared,
            "proj1":    proj1["user"],
            "proj2":    proj2["user"],
            "pred":     pred,
        }

    # ── loss computation ─────────────────────────────────────────────────

    def compute_losses(
        self,
        fwd:         Dict[str, torch.Tensor],
        split:       torch.Tensor,              # 0=train, 1=val, 2=test  (N,)
        labels:      Optional[torch.Tensor] = None,  # (N,) long  – true labels
    ) -> Dict[str, torch.Tensor]:
        
        proj1 = fwd["proj1"]
        proj2 = fwd["proj2"]
        pred  = fwd["pred"]

        device = proj1.device
        split  = split.to(device)

        # ── Loss-1: contrastive ───────────────────────────────────────────
        l_cl = unsupervised_cl_loss(proj1, proj2, split, self.cacl_cfg.cl_tau)

        # ── Loss-2: classification on training nodes ──────────────────────
        l_cls = torch.tensor(0.0, device=device)
        if self.use_classifier and pred is not None and labels is not None:
            train_mask = split == 0
            if train_mask.any():
                l_cls = F.cross_entropy(
                    pred[train_mask],
                    labels.to(device)[train_mask],
                )

        l_total = self.cacl_cfg.cl_alpha * l_cl + self.cacl_cfg.cl_beta * l_cls

        return {
            "loss":           l_total,
            "contrastive":    l_cl,
            "classification": l_cls,
        }

    # ── inference ─────────────────────────────────────────────────────────

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
      
        self.eval()
        out = self.forward(graph, num, cat, desc_emb, tweet_emb, tweet_len)

        if out["pred"] is not None:
            return out["pred"].argmax(dim=1)

        # Fallback: 2-centroid nearest-neighbour on projections
        proj = out["proj"]                          # (N, proj_out)
        centroid_0 = proj.mean(dim=0, keepdim=True) # rough centroid (unlabelled)
        # Without a classifier we simply threshold on the L2 norm from mean
        # (minimal heuristic — only used when use_classifier=False)
        dists = torch.norm(proj - centroid_0, dim=1)
        threshold = dists.median()
        return (dists > threshold).long()



@torch.no_grad()
def evaluate_cacl(
    model:     CACLBaseline,
    graph:     HeteroData,
    num:       torch.Tensor,
    cat:       torch.Tensor,
    desc_emb:  torch.Tensor,
    tweet_emb: torch.Tensor,
    tweet_len: torch.Tensor,
    labels:    torch.Tensor,   # (N,) full label tensor (may include val/test)
    mask:      Optional[torch.Tensor] = None,  # (N,) bool – which nodes to score
    device:    str = "cpu",
) -> Dict[str, float]:
    """
    Standard sklearn metrics on the selected node subset.

    No Hungarian matching is needed here because the classifier directly
    outputs class probabilities aligned with the true label space (0/1).
    This contrasts with the hybrid model's _eval(), which must remap
    unsupervised cluster ids to true labels via the Hungarian algorithm.
    """
    model.eval()
    preds = model.predict(
        graph,
        num.to(device), cat.to(device),
        desc_emb.to(device), tweet_emb.to(device), tweet_len.to(device),
    ).cpu().numpy()

    y_true = labels.cpu().numpy()

    if mask is not None:
        y_true = y_true[mask.cpu().numpy()]
        preds  = preds[mask.cpu().numpy()]

    return {
        "accuracy":  accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, average="binary", zero_division=0),
        "recall":    recall_score(y_true, preds, average="binary", zero_division=0),
        "f1":        f1_score(y_true, preds, average="binary", zero_division=0),
    }



#  Training loop


def train_cacl_baseline(
    model:      CACLBaseline,
    # raw user-feature tensors (all N nodes)
    num:        torch.Tensor,
    cat:        torch.Tensor,
    desc_emb:   torch.Tensor,
    tweet_emb:  torch.Tensor,
    tweet_len:  torch.Tensor,
    # full HeteroData graph (for CACL encoder)
    graph:      HeteroData,
    # node split: 0=train, 1=val, 2=test  shape (N,)
    split:      torch.Tensor,
    # ground-truth labels (N,) – only train nodes are used for classification loss
    labels:     Optional[torch.Tensor] = None,
    device:     str           = "cpu",
    cfg:        BaselineTrainCfg = None,
) -> Dict:

    cfg = cfg or BaselineTrainCfg()
    model.to(device)

    num       = num.to(device)
    cat       = cat.to(device)
    desc_emb  = desc_emb.to(device)
    tweet_emb = tweet_emb.to(device)
    tweet_len = tweet_len.to(device)
    split     = split.to(device)

    train_mask = split == 0
    val_mask   = split == 1

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
    )

    history: Dict[str, list] = {
        "loss":           [],
        "contrastive":    [],
        "classification": [],
        "train_accuracy": [],
        "train_f1":       [],
        "val_accuracy":   [],
        "val_f1":         [],
    }

    print("=" * 60)
    print("CACL Baseline  –  joint training")
    print(f"  epochs={cfg.epochs}  lr={cfg.lr}  wd={cfg.wd}")
    print(f"  contrastive weight α={model.cacl_cfg.cl_alpha}")
    print(f"  classification weight β={model.cacl_cfg.cl_beta}  "
          f"({'active' if (model.use_classifier and labels is not None) else 'inactive – no labels or no head'})")
    print("=" * 60)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # ── forward (two augmented views) ─────────────────────────────────
        fwd = model.forward_two_views(
            graph, num, cat, desc_emb, tweet_emb, tweet_len
        )

        # ── losses ────────────────────────────────────────────────────────
        losses = model.compute_losses(fwd, split, labels)
        losses["loss"].backward()
        optimizer.step()

        history["loss"].append(losses["loss"].item())
        history["contrastive"].append(losses["contrastive"].item())
        history["classification"].append(losses["classification"].item())

        # ── periodic evaluation ───────────────────────────────────────────
        if (ep % cfg.log_every == 0 or ep == 1) and labels is not None:
            train_metrics = evaluate_cacl(
                model, graph,
                num, cat, desc_emb, tweet_emb, tweet_len,
                labels, mask=train_mask, device=device,
            )
            val_metrics = evaluate_cacl(
                model, graph,
                num, cat, desc_emb, tweet_emb, tweet_len,
                labels, mask=val_mask if val_mask.any() else None,
                device=device,
            )
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["train_f1"].append(train_metrics["f1"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])

            print(
                f"  [{ep:04d}/{cfg.epochs}]"
                f"  total={losses['loss'].item():.4f}"
                f"  CL={losses['contrastive'].item():.4f}"
                f"  cls={losses['classification'].item():.4f}"
                f"  tr_acc={train_metrics['accuracy']:.4f}"
                f"  tr_f1={train_metrics['f1']:.4f}"
                f"  val_acc={val_metrics['accuracy']:.4f}"
                f"  val_f1={val_metrics['f1']:.4f}"
            )
        elif ep % cfg.log_every == 0 or ep == 1:
            # No labels: print loss only
            print(
                f"  [{ep:04d}/{cfg.epochs}]"
                f"  total={losses['loss'].item():.4f}"
                f"  CL={losses['contrastive'].item():.4f}"
                f"  cls={losses['classification'].item():.4f}"
            )

    # ── final evaluation ──────────────────────────────────────────────────
    final: Dict = {}
    if labels is not None:
        test_mask = split == 2
        eval_mask = test_mask if test_mask.any() else None
        final = evaluate_cacl(
            model, graph,
            num, cat, desc_emb, tweet_emb, tweet_len,
            labels, mask=eval_mask, device=device,
        )
        print("\n[CACL Baseline] Final test metrics:")
        for k, v in final.items():
            print(f"  {k:12s}: {v:.4f}")

    return {"history": history, "final": final}



#  Quick smoke-test  (python cacl_baseline.py)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    from torch_geometric.data import HeteroData

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = torch.load(
        r"D:\3rd year project\Implementation\Datasets\twibot22_smoke\graph_data_model_ready_real.pt"
    )

    num       = raw["num"]
    cat       = raw["cat"]
    desc_emb  = raw["desc_emb"]
    tweet_emb = raw["tweet_emb"]
    tweet_len = raw["tweet_len"]
    labels    = raw["labels"]
    split     = raw["split"]
    edge_index = raw["edge_index"]

    print("\n===== CACL SANITY CHECK =====")
    print("num shape      :", num.shape)
    print("cat shape      :", cat.shape)
    print("desc_emb shape :", desc_emb.shape)
    print("tweet_emb shape:", tweet_emb.shape)
    print("tweet_len shape:", tweet_len.shape)
    print("labels shape   :", labels.shape)
    print("split shape    :", split.shape)
    print("edge_index shape:", edge_index.shape)

    assert num.shape[1] == 13, f"Expected 13 numerical features, got {num.shape[1]}"
    assert cat.shape[1] == 15, f"Expected 15 categorical features, got {cat.shape[1]}"

    print("num has nan:", torch.isnan(num).any().item())
    print("cat has nan:", torch.isnan(cat).any().item())
    print("desc_emb has nan:", torch.isnan(desc_emb).any().item())
    print("tweet_emb has nan:", torch.isnan(tweet_emb).any().item())

    # minimal hetero graph
    graph = HeteroData()
    graph['user'].x = torch.zeros(num.size(0), 256)   # placeholder, overwritten by X_shared
    graph[('user', 'follows', 'user')].edge_index = edge_index

    metadata = (graph.node_types, graph.edge_types)

    feat_cfg = FeatureEncoderCfg(
        num_features=13,
        cat_features=15,
        roberta_dim=768,
        lstm_hidden=128,
        embed_dim=256,
    )

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
        in_channels={'user': feat_cfg.embed_dim},
        feat_cfg=feat_cfg,
        cacl_cfg=cacl_cfg,
        use_classifier=True,
    )

    train_cfg = BaselineTrainCfg(
        epochs=1,
        log_every=1,
    )

    result = train_cacl_baseline(
        model,
        num, cat, desc_emb, tweet_emb, tweet_len,
        graph, split, labels=labels,
        device=device, cfg=train_cfg,
    )

    print("\n[smoke-test] history keys :", list(result["history"].keys()))
    print("[smoke-test] final keys   :", list(result["final"].keys()))
    print("[smoke-test] PASSED ✓")