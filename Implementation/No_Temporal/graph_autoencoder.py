"""
graph_autoencoder.py  –  sparse
========================================

botdcgcpipeline.py receives these four return values identically —
no changes needed to forward_from_shared_features or compute_losses calls.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as _pyg_softmax


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-group softmax. Uses PyG's built-in — no torch_scatter required."""
    return _pyg_softmax(src, index, num_nodes=dim_size)


def _scatter_add_1d(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index, src)
    return out


def _scatter_add_2d(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    return out


#  Sparse graph utilities

def make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Add reverse edges, deduplicate."""
    rev = edge_index.flip(0)
    ei  = torch.cat([edge_index, rev], dim=1)
    ei  = torch.unique(ei, dim=1)
    return ei


def add_self_loops_to(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    self_loops = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).expand(2, -1)
    ei = torch.cat([edge_index, self_loops], dim=1)
    return torch.unique(ei, dim=1)


def sparse_row_normalise(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Row-normalise edge weights: w_ij  →  w_ij / Σ_k w_ik."""
    src = edge_index[0]
    deg = _scatter_add_1d(edge_weight, src, num_nodes).clamp(min=1e-8)
    return edge_weight / deg[src]


def build_sparse_transition(
    edge_index: torch.Tensor,
    num_nodes:  int,
    undirected: bool = True,
    add_self:   bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (edge_index, B_ij) where B is the row-normalised transition matrix.
    Only non-zero entries are stored  →  O(E) memory.
    """
    ei = edge_index.clone()
    if undirected:
        ei = make_undirected(ei)
    if add_self:
        ei = add_self_loops_to(ei, num_nodes)

    w = torch.ones(ei.size(1), dtype=torch.float32, device=ei.device)
    w = sparse_row_normalise(ei, w, num_nodes)
    return ei, w


def build_sparse_extended_adjacency(
    edge_index: torch.Tensor,
    num_nodes:  int,
    order_t:    int,
    undirected: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # Build B as a sparse COO tensor
    B_ei, B_w = build_sparse_transition(edge_index, num_nodes, undirected=undirected)

    B_sparse = torch.sparse_coo_tensor(
        B_ei, B_w, (num_nodes, num_nodes),
        dtype=torch.float32, device=edge_index.device,
    ).coalesce()

    power  = B_sparse
    M_sparse = torch.zeros_like(B_sparse)    # sparse + sparse = sparse

    for _ in range(order_t):
        M_sparse = (M_sparse + power).coalesce()
        if _ < order_t - 1:
            power = torch.sparse.mm(power, B_sparse).coalesce()

    M_sparse = (M_sparse / float(order_t)).coalesce()

    M_ei = M_sparse.indices()    # (2, E_M)
    M_w  = M_sparse.values()     # (E_M,)

    return M_ei, M_w


def sample_negative_edges(
    pos_edge_index: torch.Tensor,
    num_nodes:      int,
    num_neg:        int,
) -> torch.Tensor:
    pos_set = set(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))
    device  = pos_edge_index.device
    neg     = []
    max_attempts = num_neg * 10

    for _ in range(max_attempts):
        if len(neg) >= num_neg:
            break
        src = torch.randint(0, num_nodes, (num_neg,))
        dst = torch.randint(0, num_nodes, (num_neg,))
        for s, d in zip(src.tolist(), dst.tolist()):
            if s != d and (s, d) not in pos_set:
                neg.append((s, d))
                if len(neg) >= num_neg:
                    break

    if not neg:
        # fallback: just return empty
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    neg_t = torch.tensor(neg, dtype=torch.long, device=device).t()
    return neg_t


#  4.2.1  Sparse GAT layer  — O(E) memory

class SparseStructuralGATLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, alpha: float = 0.2,
                 dropout: float = 0.1):
        super().__init__()
        self.W        = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Linear(out_dim, 1, bias=False)
        self.attn_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky    = nn.LeakyReLU(alpha)
        self.norm     = nn.LayerNorm(out_dim)
        self.act      = nn.ELU()
        self.drop     = nn.Dropout(dropout)
        self.residual = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self,
        X:    torch.Tensor,
        M_ei: torch.Tensor,
        M_w:  torch.Tensor,
    ) -> torch.Tensor:
        N       = X.size(0)
        H       = self.W(X)
        res     = self.residual(X)      # skip path

        src_idx = M_ei[0]
        dst_idx = M_ei[1]

        e_src  = self.attn_src(H).squeeze(-1)
        e_dst  = self.attn_dst(H).squeeze(-1)
        e_ij   = self.leaky(e_src[src_idx] + e_dst[dst_idx])
        logits = M_w * e_ij

        alpha  = _scatter_softmax(logits, src_idx, N)
        alpha  = self.drop(alpha)

        msg = H[dst_idx] * alpha.unsqueeze(-1)
        Z   = _scatter_add_2d(msg, src_idx, N)

        has_nb = torch.zeros(N, dtype=torch.bool, device=X.device)
        has_nb[src_idx] = True
        Z[~has_nb] = H[~has_nb]

        # Residual + LayerNorm + ELU
        Z = self.norm(Z + res)
        Z = self.act(Z)
        return Z


class SparseGraphAttentionEncoder(nn.Module):
    """Stacks multiple SparseStructuralGATLayer layers."""

    def __init__(self, in_dim: int, hidden_dims: List[int], alpha: float = 0.2,
                 dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + hidden_dims
        self.layers = nn.ModuleList([
            SparseStructuralGATLayer(dims[i], dims[i + 1], alpha=alpha,
                                     dropout=dropout)
            for i in range(len(dims) - 1)
        ])

    def forward(
        self,
        X:    torch.Tensor,
        M_ei: torch.Tensor,
        M_w:  torch.Tensor,
    ) -> torch.Tensor:
        Z = X
        for layer in self.layers:
            Z = layer(Z, M_ei, M_w)
        return Z



#  4.2.2  Sparse decoder  — edge-level scores only


def edge_dot_scores(Z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    For each edge (i,j), compute σ(zᵢ · zⱼ).
    Returns shape (E,).  O(E × D) memory.
    """
    return torch.sigmoid((Z[edge_index[0]] * Z[edge_index[1]]).sum(dim=-1))



#  4.2.3  Sparse reconstruction loss


def reconstruction_loss(
    A_true: object,       # kept for API compat — ignored in sparse mode
    A_hat:  object,       # (scores, pos_ei, neg_ei) tuple
    eps:    float = 1e-8,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:

    if isinstance(A_hat, tuple):
        pos_scores, neg_scores, pos_ei = A_hat
    else:
        # Legacy dense path — kept so external callers with dense tensors
        # can still use this function directly.
        A_true_t = A_true.float() if hasattr(A_true, 'float') else A_true
        pos = A_true_t.sum().clamp(min=1.0)
        neg = A_true_t.numel() - pos
        pw  = (neg / pos).item() if pos_weight is None else pos_weight
        w   = torch.ones_like(A_true_t)
        w[A_true_t > 0] = pw
        return F.binary_cross_entropy(
            A_hat.clamp(eps, 1 - eps), A_true_t, weight=w
        )

    ones  = torch.ones_like(pos_scores)
    zeros = torch.zeros_like(neg_scores)

    if pos_weight is None:
        n_pos = pos_scores.numel()
        n_neg = neg_scores.numel()
        pos_weight = float(n_neg) / float(n_pos + 1e-8)

    pw_tensor = torch.tensor(pos_weight, dtype=pos_scores.dtype,
                             device=pos_scores.device)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores.logit().clamp(-10, 10), ones,
        pos_weight=pw_tensor,
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores.logit().clamp(-10, 10), zeros,
    )
    return (pos_loss + neg_loss) / 2.0



#  4.2.4  Sparse contrastive loss


def graph_structure_contrastive_loss(
    Z:    torch.Tensor,
    M:    object,          # (M_ei, M_w) tuple from forward()
    tau:  float = 1.0,
    eps:  float = 1e-8,
) -> torch.Tensor:

    if not isinstance(M, tuple):
        # Legacy dense path
        N   = Z.size(0)
        sim = _pairwise_cosine(Z)
        exp_sim = torch.exp(sim / tau)
        eye      = torch.eye(N, device=Z.device, dtype=torch.bool)
        pos_mask = (M > 0) & (~eye)
        neg_mask = (~pos_mask) & (~eye)
        pos_w    = torch.where(pos_mask, M, torch.zeros_like(M))
        neg_w    = neg_mask.float()
        pos_term = (pos_w * exp_sim).sum(1)
        denom    = pos_term + (neg_w * exp_sim).sum(1)
        return (-torch.log((pos_term + eps) / (denom + eps))).mean()

    M_ei, M_w = M
    N = Z.size(0)
    device = Z.device

    # ── positive terms (edges in M) ────────────────────────────────────
    src_i = M_ei[0]    # (E_M,)
    dst_j = M_ei[1]    # (E_M,)

    Z_norm = F.normalize(Z, p=2, dim=1)           # (N, D) normalised
    sim_pos = (Z_norm[src_i] * Z_norm[dst_j]).sum(-1)    # (E_M,)
    exp_pos = torch.exp(sim_pos / tau)             # (E_M,)
    pos_term_per_node = _scatter_add_1d(
        M_w * exp_pos, src_i, N
    )                                              # (N,)

    
    n_neg = min(N * 5, 10_000)    # ~5 negatives per node, max 10k total
    neg_ei = sample_negative_edges(M_ei, N, n_neg)

    if neg_ei.size(1) > 0:
        neg_src = neg_ei[0]
        neg_dst = neg_ei[1]
        sim_neg = (Z_norm[neg_src] * Z_norm[neg_dst]).sum(-1)
        exp_neg = torch.exp(sim_neg / tau)
        neg_term_per_node = _scatter_add_1d(
            exp_neg, neg_src, N
        )                                          # (N,)
    else:
        neg_term_per_node = torch.zeros(N, device=device)

    denom    = pos_term_per_node + neg_term_per_node + eps
    loss_per = -torch.log((pos_term_per_node + eps) / denom)

    # Only average over nodes that actually have neighbours in M
    has_pos = torch.zeros(N, dtype=torch.bool, device=device)
    has_pos[src_i] = True
    if has_pos.any():
        return loss_per[has_pos].mean()
    return loss_per.mean()


def _pairwise_cosine(Z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    Z_norm = Z / Z.norm(dim=1, keepdim=True).clamp(min=eps)
    return Z_norm @ Z_norm.t()


#  Full Section 4.2 module  —  public API unchanged


class BotDCGCGraphAutoEncoder(nn.Module):
    

    def __init__(
        self,
        in_dim:          int,
        gat_hidden_dims: List[int],
        order_t:         int   = 10,
        alpha:           float = 0.2,
        tau:             float = 0.5,    # lowered from 1.0 — sharpens contrast
        undirected:      bool  = True,
        add_self_loops:  bool  = False,
        neg_sample_mult: int   = 2,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.order_t        = order_t
        self.tau            = tau
        self.undirected     = undirected
        self.add_self_loops = add_self_loops
        self.neg_sample_mult = neg_sample_mult

        self.encoder = SparseGraphAttentionEncoder(
            in_dim=in_dim,
            hidden_dims=gat_hidden_dims,
            alpha=alpha,
            dropout=dropout,
        )
        # decoder is now just the edge_dot_scores function (no parameters)

    def forward(
        self,
        X:          torch.Tensor,   # (N, D)
        edge_index: torch.Tensor,   # (2, E)
    ) -> Tuple[torch.Tensor, object, torch.Tensor, object]:
        """
        Returns
        -------
        Z      : (N, D_out)
        A_hat  : (pos_scores, neg_scores, pos_ei)  — sparse recon output
        A      : (2, E)  positive edge index
        M      : (M_ei, M_w)  sparse topological relevance
        """
        N = X.size(0)

        # Build sparse extended adjacency M
        M_ei, M_w = build_sparse_extended_adjacency(
            edge_index, N, self.order_t, undirected=self.undirected
        )

        # Sparse GAT encoder
        Z = self.encoder(X, M_ei, M_w)

        # Decoder: score positive edges
        pos_ei     = M_ei                         # use M edges as positives
        pos_scores = edge_dot_scores(Z, pos_ei)   # (E_M,)

        # Sample negative edges
        n_neg = max(pos_ei.size(1) * self.neg_sample_mult, 64)
        neg_ei     = sample_negative_edges(pos_ei, N, n_neg)
        if neg_ei.size(1) > 0:
            neg_scores = edge_dot_scores(Z, neg_ei)
        else:
            neg_scores = torch.zeros(0, device=X.device)

        A_hat = (pos_scores, neg_scores, pos_ei)
        A     = edge_index          # original edge index (not M-expanded)
        M     = (M_ei, M_w)

        return Z, A_hat, A, M

    def compute_losses(
        self,
        Z:     torch.Tensor,
        A_hat: object,
        A:     object,
        M:     object,
        recon_pos_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        4.2.3  reconstruction loss  (sparse)
        4.2.4  contrastive loss     (sparse)
        """
        l_recon   = reconstruction_loss(A, A_hat, pos_weight=recon_pos_weight)
        l_contrast = graph_structure_contrastive_loss(Z, M, tau=self.tau)
        return l_recon, l_contrast



#  Legacy dense helpers kept for any external callers


def build_binary_adjacency(
    edge_index: torch.Tensor,
    num_nodes:  int,
    undirected: bool = True,
    add_self_loops: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Dense N×N adjacency — only safe for small N (<= 2000)."""
    if device is None:
        device = edge_index.device
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    if undirected:
        A[edge_index[1], edge_index[0]] = 1.0
    if add_self_loops:
        A.fill_diagonal_(1.0)
    return A


def build_transition_matrix(A: torch.Tensor) -> torch.Tensor:
    deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A / deg


def build_extended_adjacency(A: torch.Tensor, order_t: int) -> torch.Tensor:
    """Dense version — kept for backward compat with small-N tests."""
    B = build_transition_matrix(A)
    power = B.clone()
    M = torch.zeros_like(B)
    for _ in range(order_t):
        M = M + power
        power = power @ B
    return M / float(order_t)