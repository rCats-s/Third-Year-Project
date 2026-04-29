

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


# ---------------------------------------------------------------------------
# Soft assignment via Student-t distribution  
# ---------------------------------------------------------------------------

def soft_assignment(z: torch.Tensor,
                    centers: torch.Tensor) -> torch.Tensor:
   
    # ||z_i - μ_u||² for all (i, u) pairs → (N, K)
    diff = z.unsqueeze(1) - centers.unsqueeze(0)          # (N, K, D)
    dist = (diff ** 2).sum(dim=-1)                        # (N, K)
    q    = 1.0 / (1.0 + dist)                             # (N, K)
    q    = q / q.sum(dim=1, keepdim=True)                 # row-normalise
    return q


# ---------------------------------------------------------------------------
# Target distribution  
# ---------------------------------------------------------------------------

def target_distribution(Q: torch.Tensor) -> torch.Tensor:
    """
    p_iu = (q_iu² / Σ_i q_iu)  /  Σ_k (q_ik² / Σ_i q_ik)

    Squares Q and re-normalises, sharpening confident assignments.
    """
    f   = Q.sum(dim=0, keepdim=True)                     # (1, K) – cluster frequencies
    P   = (Q ** 2) / f.clamp(min=1e-9)                  # (N, K)
    P   = P / P.sum(dim=1, keepdim=True)                 # row-normalise
    return P


# ---------------------------------------------------------------------------
# Clustering loss  
# ---------------------------------------------------------------------------

def clustering_loss(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    L_clustering = KL(P || Q) = Σ_i Σ_u  p_iu · log(p_iu / q_iu)
    """
    eps = 1e-9
    loss = (P * torch.log((P + eps) / (Q + eps))).sum(dim=1).mean()
    return loss

# ---------------------------------------------------------------------------
# Final total loss  (Eq. 21)
# ---------------------------------------------------------------------------

def total_loss(l_recon: torch.Tensor,
               l_contrast: torch.Tensor,
               l_cluster: torch.Tensor,
               gamma1: float = 2.0,
               gamma2: float = 10.0) -> torch.Tensor:
    """
    Eq. (21):
        L = L_reconstruction + gamma1 * L_contrastive + gamma2 * L_clustering
    """
    return l_recon + gamma1 * l_contrast + gamma2 * l_cluster

# ---------------------------------------------------------------------------
# ClusteringModule
# ---------------------------------------------------------------------------

class ClusteringModule(nn.Module):
    """
    Holds learnable cluster centres μ and exposes Q / P computation.

    Centres are initialised with K-means on the initial node embeddings
    (call `initialise(z)` before the main training loop).
    """
    def __init__(self, n_clusters: int = 2, embed_dim: int = 16):
        super().__init__()
        self.K = n_clusters
        # Learnable cluster centres  (K, D)
        self.centers = nn.Parameter(
            torch.randn(n_clusters, embed_dim), requires_grad=True
        )

    @torch.no_grad()
    def initialise(self, z: torch.Tensor, random_state: int = 42):
        """
        Run K-means on embeddings z (numpy-compatible) and load the
        found centroids as the initial cluster centres.
        """
        z_np = z.detach().cpu().numpy()
        km   = KMeans(n_clusters=self.K, n_init=20, random_state=random_state)
        km.fit(z_np)
        centers = torch.tensor(km.cluster_centers_,
                               dtype=torch.float32,
                               device=z.device)
        self.centers.data.copy_(centers)
        print(f"[ClusteringModule] K-means initialised. "
              f"Inertia = {km.inertia_:.4f}")

    def forward(self, z: torch.Tensor):
        """
        Returns:
            Q : soft assignments (Eq. 18)
            P : target distribution (Eq. 19)
        """
        Q = soft_assignment(z, self.centers)
        P = target_distribution(Q)
        return Q, P

    def get_P(self, Q: torch.Tensor) -> torch.Tensor:
        return target_distribution(Q)

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Hard cluster labels (Eq. 22):  ŷ_i = argmax_u q_iu
        """
        Q,_ = self.forward(z)
        return Q.argmax(dim=1)