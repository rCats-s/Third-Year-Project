import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from feature_encoder import UserFeatureEncoder
from graph_autoencoder import BotDCGCGraphAutoEncoder, reconstruction_loss
from clustering import ClusteringModule, clustering_loss, total_loss
from scipy.optimize import linear_sum_assignment

def _lap(cost):
    r, c = linear_sum_assignment(cost)
    return list(zip(r, c))

base_path = r"D:\3rd year project\Implementation\Datasets\twibot22_smoke"

class BotDCGC(nn.Module):
    def __init__(
    self,
    user_embed_dim: int = 256,
    gat_hidden_dims=None,
    n_clusters: int = 2,
    order_t: int = 10,
    alpha: float = 0.2,
    tau: float = 1.0,
    gamma1: float = 2.0,
    gamma2: float = 10.0,
    undirected: bool = True,
    add_self_loops: bool = False,
    ):
        super().__init__()

        if gat_hidden_dims is None:
            gat_hidden_dims = [256, 256, 16]

        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.gae = BotDCGCGraphAutoEncoder(
            in_dim=user_embed_dim,
            gat_hidden_dims=gat_hidden_dims,
            order_t=order_t,
            alpha=alpha,
            tau=tau,
            undirected=undirected,
            add_self_loops=add_self_loops,
        )

        self.clustering = ClusteringModule(
            n_clusters=n_clusters,
            embed_dim=gat_hidden_dims[-1],
        )

    def forward_from_shared_features(self, X_shared, edge_index):
        Z, A_hat, A, M = self.gae(X_shared, edge_index)
        Q, P = self.clustering(Z)

        return {
            "X": X_shared,
            "Z": Z,
            "A_hat": A_hat,
            "A": A,
            "M": M,
            "Q": Q,
            "P": P,
        }

    def compute_losses(self, outputs, recon_pos_weight=None):
        Z = outputs["Z"]
        A_hat = outputs["A_hat"]
        A = outputs["A"]
        M = outputs["M"]
        Q = outputs["Q"]
        P = outputs["P"]

        l_recon, l_contrast = self.gae.compute_losses(
            Z, A_hat, A, M, recon_pos_weight=recon_pos_weight
        )
        l_cluster = clustering_loss(P, Q)
        l_total = total_loss(
            l_recon=l_recon,
            l_contrast=l_contrast,
            l_cluster=l_cluster,
            gamma1=self.gamma1,
            gamma2=self.gamma2,
        )

        return {
            "loss": l_total,
            "reconstruction": l_recon,
            "contrastive": l_contrast,
            "clustering": l_cluster,
        }

    @torch.no_grad()
    def initialise_clusters_from_shared_features(self, X_shared, edge_index):
        self.eval()
        outputs = self.forward_from_shared_features(X_shared, edge_index)
        self.clustering.initialise(outputs["Z"])

    @torch.no_grad()
    def predict_from_shared_features(self, X_shared, edge_index):
        self.eval()
        outputs = self.forward_from_shared_features(X_shared, edge_index)
        return self.clustering.predict(outputs["Z"])
    

from scipy.optimize import linear_sum_assignment

def _lap(cost):
    r, c = linear_sum_assignment(cost)
    return list(zip(r, c))


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int = 2) -> Tuple[float, np.ndarray]:
    n_classes = max(n_clusters, int(y_true.max()) + 1)
    D = np.zeros((n_clusters, n_classes), dtype=np.int64)

    for pred, true in zip(y_pred, y_true):
        D[pred, true] += 1

    pairs = _lap(-D)
    mapping = {p: t for p, t in pairs}
    y_remap = np.array([mapping.get(p, p) for p in y_pred])

    acc = accuracy_score(y_true, y_remap)
    return acc, y_remap


def evaluate(shared_feature_encoder, model, dataset, device):
    model.eval()
    shared_feature_encoder.eval()

    with torch.no_grad():
        X_shared = shared_feature_encoder(
            dataset.num.to(device),
            dataset.cat.to(device),
            dataset.desc_emb.to(device),
            dataset.tweet_emb.to(device),
            dataset.tweet_len.to(device),
        )

        preds = model.predict_from_shared_features(
            X_shared,
            dataset.edge_index.to(device),
        ).cpu().numpy()

    mask = dataset.labels.cpu().numpy() >= 0
    y_true = dataset.labels.cpu().numpy()[mask]
    y_pred = preds[mask]

    _, y_remap = cluster_accuracy(y_true, y_pred, n_clusters=model.clustering.K)

    return {
        "accuracy": accuracy_score(y_true, y_remap),
        "precision": precision_score(y_true, y_remap, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_remap, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_remap, average="binary", zero_division=0),
    }

def train(
    shared_feature_encoder,
    model,
    dataset,
    device,
    epochs: int = 200,
    lr: float = 5e-3,
    wd: float = 5e-3,
    pretrain_epochs: int = 30,
    log_every: int = 10,
    save_path: str = None,
):
    shared_feature_encoder.to(device)
    model.to(device)

    num = dataset.num.to(device)
    cat = dataset.cat.to(device)
    desc_emb = dataset.desc_emb.to(device)
    tweet_emb = dataset.tweet_emb.to(device)
    tweet_len = dataset.tweet_len.to(device)
    print("\n===== SANITY CHECK =====")

    print("num shape      :", num.shape)
    print("cat shape      :", cat.shape)
    print("desc_emb shape :", desc_emb.shape)
    print("tweet_emb shape:", tweet_emb.shape)
    print("tweet_len shape:", tweet_len.shape)

    # EXPECTED SHAPES
    assert num.shape[1] == 13, f"❌ Expected 13 numerical features, got {num.shape[1]}"
    assert cat.shape[1] == 15, f"❌ Expected 15 categorical features, got {cat.shape[1]}"

    # CHECK FOR BAD VALUES
    print("num has nan:", torch.isnan(num).any().item())
    print("cat has nan:", torch.isnan(cat).any().item())
    print("num has inf:", torch.isinf(num).any().item())
    print("cat has inf:", torch.isinf(cat).any().item())

    # SAMPLE VALUES
    print("\nFirst 3 num rows:")
    print(num[:3])

    print("\nFirst 3 cat rows:")
    print(cat[:3])
    edge_index = dataset.edge_index.to(device)


    optimizer = optim.Adam(
        list(shared_feature_encoder.parameters()) + list(model.parameters()),
        lr=lr,
        weight_decay=wd
    )

    print(f"[train] Pretraining graph autoencoder for {pretrain_epochs} epochs...")
    for ep in range(1, pretrain_epochs + 1):
        shared_feature_encoder.train()
        model.train()
        optimizer.zero_grad()

        X_shared = shared_feature_encoder(num, cat, desc_emb, tweet_emb, tweet_len)
        outputs = model.forward_from_shared_features(X_shared, edge_index)
        l_recon = reconstruction_loss(outputs["A"], outputs["A_hat"])

        l_recon.backward()
        optimizer.step()

        if ep % 10 == 0 or ep == 1:
            print(f"  pretrain [{ep:03d}/{pretrain_epochs}]  L_recon={l_recon.item():.4f}")

    print("[train] Initialising cluster centres with K-means...")
    with torch.no_grad():
        X_shared = shared_feature_encoder(num, cat, desc_emb, tweet_emb, tweet_len)
    model.initialise_clusters_from_shared_features(X_shared, edge_index)

    print(f"[train] Main training for {epochs} epochs...")

    history = {
        "loss": [],
        "reconstruction": [],
        "contrastive": [],
        "clustering": [],
        "accuracy": [],
        "f1": [],
    }

    for ep in range(1, epochs + 1):
        shared_feature_encoder.train()
        model.train()
        optimizer.zero_grad()

        X_shared = shared_feature_encoder(num, cat, desc_emb, tweet_emb, tweet_len)
        outputs = model.forward_from_shared_features(X_shared, edge_index)
        losses = model.compute_losses(outputs)

        losses["loss"].backward()
        optimizer.step()

        for k, v in losses.items():
            history[k].append(v.item())

        if ep % log_every == 0 or ep == 1:
            metrics = evaluate(shared_feature_encoder, model, dataset, device)
            history["accuracy"].append(metrics["accuracy"])
            history["f1"].append(metrics["f1"])

            print(
                f"  epoch [{ep:03d}/{epochs}] "
                f"loss={losses['loss'].item():.4f} "
                f"rec={losses['reconstruction'].item():.4f} "
                f"con={losses['contrastive'].item():.4f} "
                f"clu={losses['clustering'].item():.4f} "
                f"acc={metrics['accuracy']:.4f} "
                f"f1={metrics['f1']:.4f}"
            )

    final = evaluate(shared_feature_encoder, model, dataset, device)
    print("\n[train] Final metrics:")
    for k, v in final.items():
        print(f"  {k:10s}: {v:.4f}")

    if save_path:
        torch.save({
            "shared_feature_encoder": shared_feature_encoder.state_dict(),
            "botdcgc_branch": model.state_dict(),
        }, save_path)
        print(f"[train] Model saved to {save_path}")

    return {"history": history, "final": final}

class SimpleDataset:
    def __init__(self, d):
        self.user_ids = d["user_ids"]
        self.labels = d["labels"]
        self.split = d["split"]
        self.edge_index = d["edge_index"]

        self.num = d["num"]
        self.cat = d["cat"]
        self.desc_emb = d["desc_emb"]
        self.tweet_emb = d["tweet_emb"]
        self.tweet_len = d["tweet_len"]
raw = torch.load(
    r"D:\3rd year project\Implementation\Datasets\twibot22_smoke\graph_data_model_ready_real.pt"
)
dataset = SimpleDataset(raw)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = torch.load(
        r"D:\3rd year project\Implementation\Datasets\twibot22_smoke\graph_data_model_ready_real.pt"
    )
    dataset = SimpleDataset(raw)

    shared_feature_encoder = UserFeatureEncoder(
        num_features=13,
        cat_features=15,
        roberta_dim=768,
        lstm_hidden=128,
        embed_dim=256,
    )

    model = BotDCGC(
        user_embed_dim=256,
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

    results = train(
        shared_feature_encoder=shared_feature_encoder,
        model=model,
        dataset=dataset,
        device=device,
        epochs=1,
        pretrain_epochs=1,
        log_every=1,
        save_path=None,
    )