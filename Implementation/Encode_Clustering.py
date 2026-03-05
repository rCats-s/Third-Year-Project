import torch
import json
import os
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from sklearn.cluster import KMeans
from torch_geometric.utils import to_dense_adj

# 1. Define paths
base_path = r"D:\3rd year project\Implementation\archive"
processed_path = os.path.join(base_path, "processed_data.pt")
id_map_path = os.path.join(base_path, "id_map.json")

# 2. Load the data
if os.path.exists(processed_path):
    print("Loading pre-processed graph data...")
    data = torch.load(processed_path, weights_only=False)
    
    with open(id_map_path, 'r') as f:
        id_map = json.load(f)
        
    print(f"Successfully loaded {data.num_nodes} nodes and {data.edge_index.shape[1]} edges.")
else:
    print("Error: Processed data not found. Please run Phase 1 first.")

# ---------------------------------------------------------
# PHASE 2: BUILDING THE ENCODERS
# ---------------------------------------------------------

# 1. MULBOT Encoder: Processes the 20x4 tweet sequences
class MulbotEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, out_dim=32):
        super(MulbotEncoder, self).__init__()
        # Bi-GRU: looks at the sequence forward and backward
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Linear layer to compress the Bi-GRU output (hidden_dim * 2) to out_dim
        self.fc = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x_seq):
        # x_seq shape: [Batch_Size, 20, 4]
        _, hn = self.rnn(x_seq) 
        # Concatenate the final hidden states of both directions
        out = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        return self.fc(out)

# 2. Graph Encoder: Processes the 15 static features + Graph structure
class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 3. COMPLETE CACL MODEL (Fusion Layer)
class CACL_FusionEncoder(nn.Module):
    def __init__(self, static_dim, seq_dim, hidden_dim, z_dim):
        super(CACL_FusionEncoder, self).__init__()
        self.graph_encoder = GraphEncoder(static_dim, hidden_dim, z_dim)
        self.mulbot_encoder = MulbotEncoder(seq_dim, hidden_dim, z_dim)
        
        # Fusion Layer: Combines two Z_dim vectors into one
        # Here we use concatenation [z_graph, z_mulbot] = 32 + 32 = 64
        self.fusion = nn.Linear(z_dim * 2, z_dim)

    def forward(self, x_static, edge_index, x_seq):
        # Generate two separate embeddings
        z_graph = self.graph_encoder(x_static, edge_index)
        z_mulbot = self.mulbot_encoder(x_seq)
        
        # Concatenate and project to final Embedding Z
        combined = torch.cat([z_graph, z_mulbot], dim=1)
        z_final = self.fusion(combined)
        
        return z_final

full_model = CACL_FusionEncoder(static_dim=15, seq_dim=4, hidden_dim=64, z_dim=32)

full_model.eval()
with torch.no_grad():
    # Pass everything: static features, edges, and tweet sequences
    z = full_model(data.x, data.edge_index, data.x_sequence)

print(f"Final Fused Embedding Shape (Z): {z.shape}")
# Target Output: torch.Size([8278, 32])
embedding_path = os.path.join(base_path, "initial_embeddings_z.pt")
print(f"\nSaving initial embeddings Z to {embedding_path}...")
torch.save(z, embedding_path)
import pandas as pd
df = pd.DataFrame(z.numpy())
df.to_csv(os.path.join(base_path, "z_inspect.csv"), index=False)

print("Phase 2 Complete: Encoders built and initial embeddings stored.")

#PHASE 3: DECODER, CLUSTERING, AND TRAINING LOOP
#THE DECODER & CLUSTERING (From Phase 3)

class InnerProductDecoder(nn.Module):
    def forward(self, z):
        # Reconstructs the user relationships [cite: 261]
        return torch.sigmoid(torch.matmul(z, z.t()))
    
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


# THE MASTER BOTDCGC MODEL

class MasterBotDCGC(nn.Module):
    def __init__(self, static_dim=15, seq_dim=4, hidden_dim=64, z_dim=32, n_clusters=2):
        super().__init__()
        # 1. Use your custom Fusion Model instead of a basic GAT
        self.encoder = CACL_FusionEncoder(static_dim, seq_dim, hidden_dim, z_dim)
        self.decoder = InnerProductDecoder()
        self.clustering = DeepClustering(n_clusters, z_dim)

    def forward(self, x_static, edge_index, x_seq):
        # The data passes through the whole pipeline
        z = self.encoder(x_static, edge_index, x_seq)
        adj_recon = self.decoder(z)
        q = self.clustering(z)
        return z, adj_recon, q

# LOSS FUNCTIONS & ADJACENCY MATRIX


def compute_extended_adj(edge_index, num_nodes, t=10):
    print(f"Computing {t}-order expanded adjacency matrix M...")
    # 1. Convert edge_index to dense Adjacency Matrix A
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    
    # 2. Convert to Transition Matrix B (row normalized)
    deg = A.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1.0 # Prevent division by zero
    B = A / deg
    
    # 3. Expand to t-orders
    M = B.clone()
    Bt = B.clone()
    for _ in range(2, t+1):
        Bt = torch.matmul(Bt, B)
        M += Bt
    return M / t

def structure_contrastive_loss(z, adj_matrix, temperature=0.5, batch_size=1024):
    z = F.normalize(z, p=2, dim=1)
    num_nodes = z.size(0)
    total_loss = 0
    for i in range(0, num_nodes, batch_size):
        end = min(i + batch_size, num_nodes)
        z_batch = z[i:end]  # [batch, 32]
        
    
        sim = torch.matmul(z_batch, z.t()) / temperature
        sim = torch.exp(sim)
        
        adj_batch = adj_matrix[i:end]
        
        positive_mask = (adj_batch > 0).float()
        negative_mask = (adj_batch == 0).float()
        
        numerator = (sim * positive_mask).sum(dim=1) + 1e-8
        denominator = (sim * negative_mask).sum(dim=1) + 1e-8
        
        batch_loss = -torch.log(numerator / (numerator + denominator))
        total_loss += batch_loss.sum()
        
    return total_loss / num_nodes


#EXECUTION & TRAINING LOOP

def train(model, data, adj_matrix, epochs=200, gamma1=2, gamma2=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-3)
    bce_loss = nn.BCELoss()
    target_adj = (adj_matrix > 0).float()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z, adj_recon, q = model(data.x, data.edge_index, data.x_sequence)
        recon_loss = bce_loss(adj_recon, target_adj)
        contrast_loss = structure_contrastive_loss(z, adj_matrix)
        p = model.clustering.target_distribution(q.detach())
        cluster_loss = F.kl_div(q.log(), p, reduction='batchmean')
        loss = recon_loss + gamma1 * contrast_loss + gamma2 * cluster_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Total Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | Cont: {contrast_loss.item():.4f} | Clust: {cluster_loss.item():.4f}")

    return model


# Prepare Adjacency Matrix (Order t=10 from Table 3 )
M = compute_extended_adj(data.edge_index, data.num_nodes, t=10)
model = MasterBotDCGC(static_dim=15, seq_dim=4, hidden_dim=64, z_dim=32, n_clusters=2)

# Warm Start the Cluster Centers
model.eval()
with torch.no_grad():
    z_init = model.encoder(data.x, data.edge_index, data.x_sequence)
model.clustering.init_centers(z_init)

# (Hyperparameters gamma1=2, gamma2=10 )
print("\nStarting Training...")
model = train(model, data, M, epochs=200, gamma1=2, gamma2=10)

#Final Predictions
model.eval()
with torch.no_grad():
    final_z, _, q = model(data.x, data.edge_index, data.x_sequence)
    pseudo_labels = torch.argmax(q, dim=1)
index_to_id = {v: k for k, v in id_map.items()}


labels_list = pseudo_labels.cpu().numpy()

# Create a list of dictionaries for easy DataFrame creation
results = []
for idx, label in enumerate(labels_list):
    results.append({
        "user_id": index_to_id[idx],
        "pseudo_label": int(label)
    })
results_df = pd.DataFrame(results)
output_path = os.path.join(base_path, "pseudo_labels.csv")
results_df.to_csv(output_path, index=False)

print(f"Successfully saved {len(results_df)} pseudo-labels to {output_path}")
human_count = (results_df['pseudo_label'] == 0).sum()
bot_count = (results_df['pseudo_label'] == 1).sum()
print(f"Summary: Cluster 0: {human_count} users | Cluster 1: {bot_count} users")
print(f"\nTraining Complete! Pseudo-labels generated for {len(pseudo_labels)} users.")
torch.save(model.state_dict(), os.path.join(base_path, "phase3_model_weights.pt"))
print("Model weights saved for Phase 4.")
