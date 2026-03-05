import torch_geometric.utils as pyg_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import os
import json
import pandas as pd
from torch_geometric.nn import SAGEConv

base_path = r"D:\3rd year project\Implementation\archive"
processed_path = os.path.join(base_path, "processed_data.pt")
weights_path = os.path.join(base_path, "phase3_model_weights.pt")
pseudo_labels_path = os.path.join(base_path, "pseudo_labels.csv")

print("Loading data and pseudo-labels...")
data = torch.load(processed_path, weights_only=False)
labels_df = pd.read_csv(pseudo_labels_path)

pseudo_labels = torch.tensor(labels_df['pseudo_label'].values, dtype=torch.long)

# RE-DEFINE ARCHITECTURE (Required for loading weights)

class MulbotEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, out_dim=32):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, out_dim)
    def forward(self, x_seq):
        _, hn = self.rnn(x_seq)
        out = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        return self.fc(out)

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index)

class CACL_FusionEncoder(nn.Module):
    def __init__(self, static_dim=15, seq_dim=4, hidden_dim=64, z_dim=32):
        super().__init__()
        self.graph_encoder = GraphEncoder(static_dim, hidden_dim, z_dim)
        self.mulbot_encoder = MulbotEncoder(seq_dim, hidden_dim, z_dim)
        self.fusion = nn.Linear(z_dim * 2, z_dim)
    def forward(self, x_static, edge_index, x_seq):
        z_graph = self.graph_encoder(x_static, edge_index)
        z_mulbot = self.mulbot_encoder(x_seq)
        return self.fusion(torch.cat([z_graph, z_mulbot], dim=1))
    
class MasterBotDCGC(nn.Module):
    def __init__(self, static_dim=15, seq_dim=4, hidden_dim=64, z_dim=32):
        super().__init__()
        self.encoder = CACL_FusionEncoder(static_dim, seq_dim, hidden_dim, z_dim)

# PHASE 4: CONTRASTIVE FINE-TUNING (CACL)

# Graph Augmentation Function
def augment_graph(x, edge_index, x_seq, drop_edge_rate=0.2, drop_feature_rate=0.2):
    """Creates a 'noisy' view of the graph by dropping edges and masking features."""
    # A. Drop Edges
    edge_index_aug, _ = pyg_utils.dropout_adj(edge_index, p=drop_edge_rate)
    
    # B. Mask Static Features
    drop_mask_x = torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_feature_rate
    x_aug = x.clone()
    x_aug[drop_mask_x] = 0
    
    # C. Mask Sequence Features (MULBOT)
    drop_mask_seq = torch.empty(x_seq.size(), dtype=torch.float32, device=x_seq.device).uniform_(0, 1) < drop_feature_rate
    x_seq_aug = x_seq.clone()
    x_seq_aug[drop_mask_seq] = 0
    
    return x_aug, edge_index_aug, x_seq_aug

# Cross-View Contrastive Loss (InfoNCE)
def cross_view_contrastive_loss(z1, z2, edge_index, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # similarity matrix
    sim = torch.exp(torch.mm(z1, z2.t()) / temperature)

    # node-to-node positives
    positives = sim.diag()

    # neighbor positives
    src, dst = edge_index
    neighbor_pos = torch.exp((z1[src] * z2[dst]).sum(dim=1) / temperature)

    pos_nodes = sim.diag()
    pos_neighbors = torch.exp((z1[src] * z2[dst]).sum(dim=1) / temperature)

    pos = torch.cat([pos_nodes, pos_neighbors])
    loss = -torch.log(pos / (sim.sum(dim=1).mean() + 1e-8)).mean()

    return loss

# The Final Classifier Model
class FinalBotClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=32):
        super().__init__()
        self.encoder = encoder # reuse the Phase 2/3 Fusion Encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 2) # Output: [Prob Human, Prob Bot]
        )

    def forward(self, x, edge_index, x_seq):
        z = self.encoder(x, edge_index, x_seq)
        z = F.normalize(z, dim=1)
        preds = self.classifier(z)
        return z, preds

# Phase 4 Training Loop
def train_phase4(classifier_model, data, pseudo_labels, epochs=100, alpha=0.1):
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.005, weight_decay=5e-4)
    # Standard Cross-Entropy for Classification
    ce_loss_fn = nn.CrossEntropyLoss()

    print("\n--- Starting Phase 4: Contrastive Fine-Tuning ---")
    classifier_model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Create two augmented views of the graph
        x_aug1, edge_aug1, seq_aug1 = augment_graph(data.x, data.edge_index, data.x_sequence)
        x_aug2, edge_aug2, seq_aug2 = augment_graph(data.x, data.edge_index, data.x_sequence)
        
        # Forward pass for both views
        z1, preds1 = classifier_model(x_aug1, edge_aug1, seq_aug1)
        z2, preds2 = classifier_model(x_aug2, edge_aug2, seq_aug2)
        
        # Calculate Prediction Loss (using Phase 3 Pseudo-labels as ground truth)
        # We average the loss across both views
        pred_loss = (ce_loss_fn(preds1, pseudo_labels) + ce_loss_fn(preds2, pseudo_labels)) / 2
        
        # 4. Calculate Cross-View Contrastive Loss
        cl_loss = cross_view_contrastive_loss(z1, z2, data.edge_index)
        
        # Joint Optimization (Alpha balances the two losses)
        total_loss = pred_loss + (alpha * cl_loss)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), max_norm=5.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Phase 4 - Epoch {epoch:03d} | Total: {total_loss.item():.4f} | Pred Loss: {pred_loss.item():.4f} | CL Loss: {cl_loss.item():.4f}")

    return classifier_model
# EXECUTE PHASE 4
print("Initializing model and loading weights...")
# a dummy Phase 3 model to act as a container for the weights
temp_phase3_container = MasterBotDCGC(static_dim=15, seq_dim=4, hidden_dim=64, z_dim=32)

# Add the missing components so the state_dict can be mapped (even if we don't use them)
temp_phase3_container.decoder = nn.Identity() 
temp_phase3_container.clustering = nn.Identity()

# Load the weights into the container
temp_phase3_container.load_state_dict(torch.load(weights_path), strict=False)

# Extract the trained encoder and build the classifier
classifier_model = FinalBotClassifier(encoder=temp_phase3_container.encoder)
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
ce_loss_fn = nn.CrossEntropyLoss()

print("Starting Fine-tuning...")
classifier_model.train()
for epoch in range(50):
    optimizer.zero_grad()
    
    # Views
    x1, e1, s1 = augment_graph(data.x, data.edge_index, data.x_sequence)
    x2, e2, s2 = augment_graph(data.x, data.edge_index, data.x_sequence)
    
    z1, p1 = classifier_model(x1, e1, s1)
    z2, p2 = classifier_model(x2, e2, s2)
    
    # Classification Loss (Teacher-Student)
    pred_loss = (ce_loss_fn(p1, pseudo_labels) + ce_loss_fn(p2, pseudo_labels)) / 2
    
    # Contrastive Loss (Fixed: Pass edge_index)
    cl_loss = cross_view_contrastive_loss(z1, z2, data.edge_index)
    
    loss = pred_loss + (0.1 * cl_loss)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | Total Loss: {loss.item():.4f} | Pred: {pred_loss.item():.4f} | CL: {cl_loss.item():.4f}")

# Final Prediction Save
final_output_path = os.path.join(base_path, "final_bot_predictions.csv")
classifier_model.eval()
with torch.no_grad():
    _, logits = classifier_model(data.x, data.edge_index, data.x_sequence)
    final_preds = torch.argmax(logits, dim=1)

labels_df['final_refined_label'] = final_preds.numpy()
labels_df.to_csv(final_output_path, index=False)
print(f"Refined labels saved to {final_output_path}")