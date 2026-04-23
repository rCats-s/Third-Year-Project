"""
BotDCGC – Section 4.2: Graph Attention Autoencoder
Implements:
    4.2.1 Graph attention encoder
    4.2.2 Inner product decoder
    4.2.3 Reconstruction loss
    4.2.4 Contrastive learning based on graph structure
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities for graph structure
# ---------------------------------------------------------------------------
def build_binary_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    undirected: bool = True,
    add_self_loops: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    edge_index: (2, E)
    returns A: (N, N) binary adjacency matrix
    """
    if device is None:
        device = edge_index.device

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    src, dst = edge_index[0], edge_index[1]
    A[src, dst] = 1.0

    if undirected:
        A[dst, src] = 1.0

    if add_self_loops:
        A.fill_diagonal_(1.0)

    return A


def build_transition_matrix(A: torch.Tensor) -> torch.Tensor:
    """
    B_ij = 1 / d_i if edge(i,j) exists, else 0
    row-normalised transition matrix
    """
    deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    B = A / deg
    return B


def build_extended_adjacency(A: torch.Tensor, order_t: int) -> torch.Tensor:
    """
    Implements Eq. (9):
        M = (B + B^2 + ... + B^t) / t
    where B is the transition matrix.

    A: (N, N) binary adjacency
    returns M: (N, N) dense topological relevance matrix
    """
    assert order_t >= 1, "order_t must be >= 1"

    B = build_transition_matrix(A)
    power = B.clone()
    M = torch.zeros_like(B)

    for _ in range(order_t):
        M = M + power
        power = power @ B

    M = M / float(order_t)
    return M


# ---------------------------------------------------------------------------
# 4.2.1 Graph attention encoder layer
# ---------------------------------------------------------------------------
class DenseStructuralGATLayer(nn.Module):
    """
    Variant of GAT using dense topological relevance matrix M.
    This matches the paper's idea of combining:
      - feature attention from [W x_i || W x_j]
      - masked / structure-aware attention via M_ij

    Input:
        X: (N, Fin)
        M: (N, N) extended adjacency / topological relevance matrix

    Output:
        Z: (N, Fout)
    """
    def __init__(self, in_dim: int, out_dim: int, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, X: torch.Tensor, M: torch.Tensor,
            chunk_size: int = 128) -> torch.Tensor:
        N = X.size(0)
        H = self.W(X)              # (N, Fout)
        logits = torch.full((N, N), float('-inf'), device=X.device)

        # Process chunk_size rows at a time — peak memory is chunk_size × N × 2F
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            Hi = H[start:end].unsqueeze(1).expand(-1, N, -1)   # (chunk, N, F)
            Hj = H.unsqueeze(0).expand(end - start, -1, -1)    # (chunk, N, F)
            pair = torch.cat([Hi, Hj], dim=-1)                  # (chunk, N, 2F)
            e = self.leaky_relu(self.attn(pair).squeeze(-1))    # (chunk, N)
            logits[start:end] = M[start:end] * e
            logits[start:end] = logits[start:end].masked_fill(
                M[start:end] == 0, float('-inf')
            )

        # Handle isolated nodes
        no_neighbor = ~(M > 0).any(dim=1)
        if no_neighbor.any():
            logits[no_neighbor] = float('-inf')
            idx = no_neighbor.nonzero(as_tuple=True)[0]
            logits[idx, idx] = 0.0

        alpha = F.softmax(logits, dim=1)   # (N, N)
        Z = torch.sigmoid(alpha @ H)
        return Z


class GraphAttentionEncoder(nn.Module):
    """
    Stacks multiple DenseStructuralGATLayer layers.

    Example:
        encoder = GraphAttentionEncoder(
            in_dim=256,
            hidden_dims=[256, 256, 16],
            alpha=0.2
        )
    """
    def __init__(self, in_dim: int, hidden_dims: List[int], alpha: float = 0.2):
        super().__init__()
        dims = [in_dim] + hidden_dims
        self.layers = nn.ModuleList([
            DenseStructuralGATLayer(dims[i], dims[i + 1], alpha=alpha)
            for i in range(len(dims) - 1)
        ])

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        Z = X
        for layer in self.layers:
            Z = layer(Z, M)
        return Z


# ---------------------------------------------------------------------------
# 4.2.2 Inner product decoder
# ---------------------------------------------------------------------------
class InnerProductDecoder(nn.Module):
    """
    Eq. (13):
        A_hat_ij = sigmoid(z_i^T z_j)
    """
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(Z @ Z.t())


# ---------------------------------------------------------------------------
# 4.2.3 Reconstruction loss
# ---------------------------------------------------------------------------
def reconstruction_loss(
    A_true: torch.Tensor,
    A_hat: torch.Tensor,
    eps: float = 1e-8,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    A_true = A_true.float()

    if pos_weight is None:
        pos = A_true.sum().clamp(min=1.0)
        neg = A_true.numel() - pos
        pos_weight = (neg / pos).item()

    weight = torch.ones_like(A_true)
    weight[A_true > 0] = pos_weight

    loss = F.binary_cross_entropy(
        A_hat.clamp(eps, 1 - eps),
        A_true,
        weight=weight
    )
    return loss


# ---------------------------------------------------------------------------
# 4.2.4 Contrastive learning based on graph structure
# ---------------------------------------------------------------------------
def pairwise_cosine_similarity(Z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    returns S where S_ij = cosine(z_i, z_j)
    """
    Z_norm = Z / Z.norm(dim=1, keepdim=True).clamp(min=eps)
    return Z_norm @ Z_norm.t()


def graph_structure_contrastive_loss(
    Z: torch.Tensor,
    M: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Implements Section 4.2.4 in the paper.

    The paper's equations define:
      - positive pairs: t-order neighbors (M_ij > 0), weighted by M_ij
      - negative pairs: non-neighbors (M_ik == 0), weighted by 1

    compute a stable version of the objective consistent with that idea:
      positives are weighted neighbors,
      denominator uses both positives and negatives excluding self.

    This is the closest practical implementation of Eqs. (15)-(17).
    """
    N = Z.size(0)
    sim = pairwise_cosine_similarity(Z)          # (N, N)
    exp_sim = torch.exp(sim / tau)

    eye = torch.eye(N, device=Z.device, dtype=torch.bool)
    pos_mask = (M > 0) & (~eye)
    neg_mask = (~pos_mask) & (~eye)

    # gamma_ij for positives = M_ij, else 0
    pos_weight = torch.where(pos_mask, M, torch.zeros_like(M))

    # gamma_ik for negatives = 1 if non-neighbor else 0
    neg_weight = neg_mask.float()

    pos_term = (pos_weight * exp_sim).sum(dim=1)                  # numerator
    denom = (pos_weight * exp_sim).sum(dim=1) + \
            (neg_weight * exp_sim).sum(dim=1)                     # denominator

    loss_i = -torch.log((pos_term + eps) / (denom + eps))
    return loss_i.mean()


# ---------------------------------------------------------------------------
# Full Section 4.2 module
# ---------------------------------------------------------------------------
class BotDCGCGraphAutoEncoder(nn.Module):
    """
    Full implementation of Section 4.2:
      4.2.1 Encoder
      4.2.2 Decoder
      4.2.3 Reconstruction loss
      4.2.4 Graph-structure contrastive loss

    Input:
        X         : (N, D) user attribute embedding from Section 4.1
        edge_index: (2, E)

    Output:
        Z         : (N, embedding_dim)
        A_hat     : (N, N)
        A         : (N, N)
        M         : (N, N)
    """
    def __init__(
        self,
        in_dim: int,
        gat_hidden_dims: List[int],
        order_t: int = 10,
        alpha: float = 0.2,
        tau: float = 1.0,
        undirected: bool = True,
        add_self_loops: bool = False,
    ):
        super().__init__()
        self.order_t = order_t
        self.tau = tau
        self.undirected = undirected
        self.add_self_loops = add_self_loops

        self.encoder = GraphAttentionEncoder(
            in_dim=in_dim,
            hidden_dims=gat_hidden_dims,
            alpha=alpha
        )
        self.decoder = InnerProductDecoder()

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = X.size(0)

        # Original adjacency A
        A = build_binary_adjacency(
            edge_index=edge_index,
            num_nodes=N,
            undirected=self.undirected,
            add_self_loops=self.add_self_loops,
            device=X.device,
        )

        # Extended adjacency / topological relevance M
        M = build_extended_adjacency(A, order_t=self.order_t)

        # 4.2.1 encode
        Z = self.encoder(X, M)

        # 4.2.2 decode
        A_hat = self.decoder(Z)

        return Z, A_hat, A, M

    def compute_losses(
        self,
        Z: torch.Tensor,
        A_hat: torch.Tensor,
        A: torch.Tensor,
        M: torch.Tensor,
        recon_pos_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 4.2.3
        l_recon = reconstruction_loss(A, A_hat, pos_weight=recon_pos_weight)

        # 4.2.4
        l_contrast = graph_structure_contrastive_loss(Z, M, tau=self.tau)

        return l_recon, l_contrast