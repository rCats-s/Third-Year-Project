import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
from torch_geometric.data import Data


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads))
        
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        
        self.layers.append(GATConv(hidden_dim * heads, embed_dim, heads=1))

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.elu(x)
        x = self.layers[-1](x, edge_index)
        return x
    
class InnerProductDecoder(nn.Module):
    def forward(self, z):
        adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_recon


def structure_contrastive_loss(z, adj_matrix, temperature=0.5):
    # sim: [N, N]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    sim = torch.exp(sim / temperature)

    positive_mask = (adj_matrix > 0).float()
    negative_mask = (adj_matrix == 0).float()

    # Sum across columns (dim=1) for all nodes simultaneously
    numerator = (sim * positive_mask).sum(dim=1)
    denominator = (sim * negative_mask).sum(dim=1)
    
    # Add epsilon to prevent log(0)
    loss = -torch.log(numerator / (numerator + denominator + 1e-8))
    
    return loss.mean()


class DeepClustering(nn.Module):
    def __init__(self, n_clusters, embed_dim):
        super().__init__()
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim))
    def init_centers(self, z):
        # Use K-Means on the initial embeddings to set the centers
        kmeans = KMeans(n_clusters=self.cluster_centers.shape[0], n_init=20)
        kmeans.fit(z.detach().cpu().numpy())
        
        # Overwrite the random parameter with the K-Means centers
        self.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(z.device)
        print("Cluster centers initialized via K-Means.")

    def forward(self, z):
        # Student-t distribution
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, dim=2))
        q = q / q.sum(dim=1, keepdim=True)
        return q

    def target_distribution(self, q):
        weight = (q ** 2) / torch.sum(q, dim=0)
        return (weight.t() / torch.sum(weight, dim=1)).t()

def clustering_loss(q, p):
    return F.kl_div(q.log(), p, reduction='batchmean')

class BotDCGC(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, embed_dim=16, n_clusters=2):
        super().__init__()
        
        self.encoder = GATEncoder(in_dim, hidden_dim, embed_dim)
        self.decoder = InnerProductDecoder()
        self.clustering = DeepClustering(n_clusters, embed_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_recon = self.decoder(z)
        q = self.clustering(z)
        return z, adj_recon, q

def compute_extended_adj(adj, t=10):
    B = adj / adj.sum(dim=1, keepdim=True)
    M = B.clone()
    Bt = B.clone()

    for _ in range(2, t+1):
        Bt = torch.matmul(Bt, B)
        M += Bt

    return M / t


def train(model, data, adj_matrix, epochs=200, gamma1=2, gamma2=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-3)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z, adj_recon, q = model(data.x, data.edge_index)

        # Reconstruction Loss
        recon_loss = F.binary_cross_entropy(adj_recon, adj_matrix)

        # Contrastive Loss
        contrast_loss = structure_contrastive_loss(z, adj_matrix)

        # Clustering Loss
        p = model.clustering.target_distribution(q.detach())
        cluster_loss = clustering_loss(q, p)

        loss = recon_loss + gamma1 * contrast_loss + gamma2 * cluster_loss

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model

with torch.no_grad():
    z, _, q = model(data.x, data.edge_index)
    preds = torch.argmax(q, dim=1)