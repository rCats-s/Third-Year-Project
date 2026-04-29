

from __future__ import annotations

import gc
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ── paths ──────────────────────────────────────────────────────────────────
INPUT_PATH  = Path(r"D:\3rd year project\Implementation\Datasets\twibot22_out2\graph_data.pt")
OUTPUT_PATH = Path(r"D:\3rd year project\Implementation\Datasets\twibot22_out2\graph_data_model_ready_notemp_real.pt")

USER_JSON_PATH  = INPUT_PATH.parent / "user.json"
TWEET_JSON_PATH = INPUT_PATH.parent / "tweet_0.json"
EDGE_CSV_PATH = INPUT_PATH.parent / "edge.csv"

# ── model / embedding config ───────────────────────────────────────────────
MODEL_NAME          = "roberta-base"
ROBERTA_DIM         = 768
MAX_TWEETS_PER_USER = 10
MAX_WORDS_PER_TWEET = 16
MAX_DESC_TOKENS     = 64

# ── batch sizes ────────────────────────────────────────────────────────────
# Raise these if you have more VRAM / RAM; lower them if you get OOM.
DESC_BATCH_SIZE  = 64   # users per desc batch
TWEET_BATCH_SIZE = 64   # (user, tweet-slot) pairs per tweet batch




def normalise_user_id(x) -> str:
    x = str(x)
    return x if x.startswith("u") else f"u{x}"


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def zscore_clip(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    mu  = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0,  keepdims=True) + 1e-8
    return np.clip((x - mu) / std, -clip, clip)


def posting_hour_entropy(timestamps: List[int]) -> float:
    valid = [t for t in timestamps if isinstance(t, (int, float)) and t > 0]
    if not valid:
        return 0.0
    hours  = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in valid]
    counts = np.bincount(hours, minlength=24).astype(np.float32)
    probs  = counts / max(counts.sum(), 1.0)
    probs  = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())




def load_tweets_single_pass(
    tweet_json_path: Path,
    user_ids: List[str],
    max_per_user: int,
) -> Tuple[Dict[str, List[int]], Dict[str, List[str]]]:
    wanted = set(user_ids)
    timestamps = defaultdict(list)
    texts = defaultdict(list)

    print(f"  streaming {tweet_json_path.name} …", flush=True)

    import ijson

    seen = 0
    matched = 0

    with open(tweet_json_path, "rb") as f:
        for tw in ijson.items(f, "item"):
            seen += 1
            if seen % 500000 == 0:
                print(f"    processed {seen:,} tweets | matched {matched:,}", flush=True)

            author = normalise_user_id(
                tw.get("author_id", tw.get("user", {}).get("id", ""))
            )
            if author not in wanted:
                continue

            matched += 1

            created_at = tw.get("created_at")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at)
                    timestamps[author].append(int(dt.timestamp()))
                except Exception:
                    pass

            if len(texts[author]) < max_per_user:
                text = tw.get("full_text", tw.get("text", ""))
                texts[author].append(str(text))

    print(f"  finished streaming: {seen:,} tweets scanned, {matched:,} matched")
    return dict(timestamps), dict(texts)




def compute_temporal_features(
    tweet_timestamps: Dict[str, List[int]],
    user_ids:         List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    temporal_num = np.zeros((len(user_ids), 6),  dtype=np.float32)
    temporal_cat = np.zeros((len(user_ids), 4),  dtype=np.float32)

    for i, uid in enumerate(tqdm(user_ids, desc="  temporal features", leave=False)):
        raw_ts = tweet_timestamps.get(uid, [])
        ts = sorted(int(t) for t in raw_ts if isinstance(t, (int, float)) and int(t) > 0)

        if not ts:
            continue

        if len(ts) >= 2:
            gaps     = np.diff(ts).astype(np.float32) / 3600.0
            mean_gap = float(gaps.mean())
            std_gap  = float(gaps.std())
        else:
            gaps     = np.array([], dtype=np.float32)
            mean_gap = 0.0
            std_gap  = 0.0

        unique_days          = len(set(int(t // 86400) for t in ts))
        tweets_per_active_day = safe_ratio(len(ts), unique_days)
        entropy              = posting_hour_entropy(ts)

        weekdays     = [datetime.fromtimestamp(t, tz=timezone.utc).weekday() for t in ts]
        weekend_ratio = safe_ratio(sum(1 for w in weekdays if w >= 5), len(ts))

        hours        = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts]
        night_ratio  = safe_ratio(sum(1 for h in hours if h < 6 or h >= 23), len(ts))

        temporal_num[i] = [mean_gap, std_gap, tweets_per_active_day,
                           entropy, weekend_ratio, night_ratio]

        cv = safe_ratio(std_gap, mean_gap)
        temporal_cat[i, 0] = 1.0 if night_ratio   >= 0.50 else 0.0
        temporal_cat[i, 1] = 1.0 if weekend_ratio >= 0.50 else 0.0
        temporal_cat[i, 2] = 1.0 if (len(gaps) >= 3 and cv < 0.50) else 0.0
        temporal_cat[i, 3] = 1.0 if (len(gaps) >= 3 and cv > 1.50) else 0.0

    return zscore_clip(temporal_num), temporal_cat


@torch.no_grad()
def build_desc_emb_batched(
    user_ids:    List[str],
    desc_lookup: Dict[str, str],
    tokenizer,
    model,
    device,
    batch_size:  int = DESC_BATCH_SIZE,
) -> torch.Tensor:
    
    N        = len(user_ids)
    desc_emb = torch.zeros((N, ROBERTA_DIM), dtype=torch.float32)
    texts    = [desc_lookup.get(uid, "") for uid in user_ids]

    model.eval()
    for start in tqdm(range(0, N, batch_size), desc="  desc embeddings"):
        end         = min(start + batch_size, N)
        batch_texts = texts[start:end]

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,            # pad to longest in THIS batch
            max_length=MAX_DESC_TOKENS,
            return_tensors="pt",
        )
        encoded  = {k: v.to(device) for k, v in encoded.items()}
        outputs  = model(**encoded)

        # CLS token (index 0) for every item in the batch
        cls_vecs           = outputs.last_hidden_state[:, 0].detach().cpu()  # (B, 768)
        desc_emb[start:end] = cls_vecs

    return desc_emb



@torch.no_grad()
def build_tweet_emb_batched(
    user_ids:     List[str],
    tweet_lookup: Dict[str, List[str]],
    tokenizer,
    model,
    device,
    batch_size:   int = TWEET_BATCH_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
   
    N = len(user_ids)

    # memory estimate
    bytes_needed = N * MAX_TWEETS_PER_USER * MAX_WORDS_PER_TWEET * ROBERTA_DIM * 4
    print(f"  pre-allocating tweet_emb: "
          f"{bytes_needed / 1e9:.2f} GB  "
          f"({N} × {MAX_TWEETS_PER_USER} × {MAX_WORDS_PER_TWEET} × {ROBERTA_DIM} float32)")

    tweet_emb = torch.zeros(
        (N, MAX_TWEETS_PER_USER, MAX_WORDS_PER_TWEET, ROBERTA_DIM),
        dtype=torch.float32,
    )
    tweet_len = torch.zeros((N, MAX_TWEETS_PER_USER), dtype=torch.long)

    # ── flatten all valid (user_idx, slot, text) triples ──────────────────
    items: List[Tuple[int, int, str]] = []
    for i, uid in enumerate(user_ids):
        tweets = tweet_lookup.get(uid, [])[:MAX_TWEETS_PER_USER]
        for j, text in enumerate(tweets):
            items.append((i, j, str(text)))

    if not items:
        print("  [WARN] no tweets found for any user")
        return tweet_emb, tweet_len

    print(f"  {len(items):,} (user, tweet) pairs to embed")

    # ── batched inference ─────────────────────────────────────────────────
    model.eval()
    for start in tqdm(range(0, len(items), batch_size), desc="  tweet embeddings"):
        batch    = items[start:start + batch_size]
        idxs_i   = [x[0] for x in batch]
        idxs_j   = [x[1] for x in batch]
        texts    = [x[2] for x in batch]

        encoded  = tokenizer(
            texts,
            truncation=True,
            padding=True,            # pad to longest in THIS batch
            max_length=MAX_WORDS_PER_TWEET,
            return_tensors="pt",
        )
        encoded  = {k: v.to(device) for k, v in encoded.items()}
        outputs  = model(**encoded)

        # hidden: (B, padded_seq_len, 768) — may be < MAX_WORDS_PER_TWEET
        hidden   = outputs.last_hidden_state.detach().cpu()     # (B, L, 768)
        attn_cpu = encoded["attention_mask"].detach().cpu()      # (B, L)
        L        = hidden.size(1)
        fill_len = min(L, MAX_WORDS_PER_TWEET)

        for k, (i, j) in enumerate(zip(idxs_i, idxs_j)):
            tweet_emb[i, j, :fill_len] = hidden[k, :fill_len]
            tweet_len[i, j]            = min(int(attn_cpu[k].sum().item()),
                                             MAX_WORDS_PER_TWEET)

    return tweet_emb, tweet_len




def load_user_descriptions(user_json_path: Path) -> Dict[str, str]:
    print(f"  reading {user_json_path.name} …", end=" ", flush=True)
    with open(user_json_path, "r", encoding="utf-8") as f:
        users = json.load(f)
    print(f"{len(users):,} users")

    return {
        normalise_user_id(u.get("id", "")): str(u.get("description") or "")
        for u in users
    }
def build_hetero_edges(edge_csv_path: Path, user_ids: List[str]):
    import pandas as pd

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    tweet_to_idx = {}
    list_to_idx = {}
    hashtag_to_idx = {}

    hetero_edges = defaultdict(lambda: [[], []])

    print(f"  reading hetero edges from {edge_csv_path.name} …")
    edge_df = pd.read_csv(edge_csv_path, dtype=str)

    edge_df.columns = [c.strip().lower() for c in edge_df.columns]
    edge_df = edge_df.rename(columns={
    "source_id": "source",
    "target_id": "target",
    "src": "source",
    "tgt": "target",
    "relation_type": "relation",
    })
    required_cols = {"source", "target", "relation"}
    missing = required_cols - set(edge_df.columns)
    if missing:
        raise ValueError(f"edge.csv missing columns: {missing}. Found: {edge_df.columns.tolist()}")  

    for row in tqdm(edge_df.itertuples(index=False), total=len(edge_df), desc="  hetero edges"):
        src = normalise_user_id(row.source)
        tgt = str(row.target)
        rel = str(row.relation)

        if src not in user_to_idx:
            continue

        src_idx = user_to_idx[src]

        # user-user edges
        if rel in {"following", "followers", "followed"}:
            tgt_user = normalise_user_id(tgt)
            if tgt_user not in user_to_idx:
                continue
            tgt_idx = user_to_idx[tgt_user]
            hetero_edges[("user", rel, "user")][0].append(src_idx)
            hetero_edges[("user", rel, "user")][1].append(tgt_idx)

        # user-tweet edges
        elif rel in {"post", "retweeted", "like", "quoted", "replied", "mentioned", "pinned"}:
            if tgt not in tweet_to_idx:
                tweet_to_idx[tgt] = len(tweet_to_idx)
            tgt_idx = tweet_to_idx[tgt]
            hetero_edges[("user", rel, "tweet")][0].append(src_idx)
            hetero_edges[("user", rel, "tweet")][1].append(tgt_idx)

        # user-list edges
        elif rel in {"own", "membership"}:
            if tgt not in list_to_idx:
                list_to_idx[tgt] = len(list_to_idx)
            tgt_idx = list_to_idx[tgt]
            hetero_edges[("user", rel, "list")][0].append(src_idx)
            hetero_edges[("user", rel, "list")][1].append(tgt_idx)

        # user-hashtag edges
        elif rel in {"discuss", "contain"}:
            if tgt not in hashtag_to_idx:
                hashtag_to_idx[tgt] = len(hashtag_to_idx)
            tgt_idx = hashtag_to_idx[tgt]
            hetero_edges[("user", rel, "hashtag")][0].append(src_idx)
            hetero_edges[("user", rel, "hashtag")][1].append(tgt_idx)

    hetero_edges = {
        edge_type: torch.tensor(indices, dtype=torch.long)
        for edge_type, indices in hetero_edges.items()
    }

    print("  hetero edge types:")
    for edge_type, edge_index in hetero_edges.items():
        print(f"    {edge_type}: {edge_index.shape[1]:,} edges")

    print(f"  tweet nodes   : {len(tweet_to_idx):,}")
    print(f"  list nodes    : {len(list_to_idx):,}")
    print(f"  hashtag nodes : {len(hashtag_to_idx):,}")

    return hetero_edges, tweet_to_idx, list_to_idx, hashtag_to_idx



def main():
    print(f"Loading graph_data.pt …  {INPUT_PATH}")
    data = torch.load(INPUT_PATH, weights_only=False)

    user_ids   = data["user_ids"]          # list[str]
    labels     = data["labels"]
    split      = data["split"]
    edge_index = data["edge_index"]

    user_num_feat = data["user_num_feat"].cpu().numpy()   # (N, F_n)
    user_cat_feat = data["user_cat_feat"].cpu().numpy()   # (N, F_c)

    N = len(user_ids)
    print(f"\n  users: {N:,}   edges: {edge_index.shape[1]:,}")

    # ── Stage 1: single JSON pass ─────────────────────────────────────────
    print("\n[Stage 1] Loading tweet JSON (single pass) …")
    tweet_timestamps, tweet_texts = load_tweets_single_pass(
        TWEET_JSON_PATH, user_ids, MAX_TWEETS_PER_USER
    )

    users_with_tw = sum(1 for xs in tweet_texts.values() if xs)
    print(f"  users with tweet texts: {users_with_tw:,} / {N:,}")

    # ── Stage 2: SKIPPED for non-temporal version ─────────────────────────
    print("\n[Stage 2] Using static features only (non-temporal version)")
    num = torch.tensor(user_num_feat, dtype=torch.float32)   # (N, 7)
    cat = torch.tensor(user_cat_feat, dtype=torch.float32)   # (N, 11)
    print(f"  num shape: {tuple(num.shape)}   cat shape: {tuple(cat.shape)}")
    # ── Stage 3: RoBERTa ──────────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Stage 3] Loading RoBERTa ({MODEL_NAME}) on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    roberta   = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # 3a: descriptions
    print("\n[Stage 3a] Description embeddings …")
    desc_lookup = load_user_descriptions(USER_JSON_PATH)
    desc_emb    = build_desc_emb_batched(
        user_ids, desc_lookup, tokenizer, roberta, device
    )

    # free GPU cache between desc and tweet passes
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # 3b: tweets
    print("\n[Stage 3b] Tweet embeddings …")
    tweet_emb, tweet_len = build_tweet_emb_batched(
        user_ids, tweet_texts, tokenizer, roberta, device
    )

    # free model weights — not needed after encoding
    del roberta
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ── Diagnostics ───────────────────────────────────────────────────────
    print("\n===== DESC EMB DEBUG =====")
    print("  shape :", tuple(desc_emb.shape))
    print("  mean  :", round(desc_emb.mean().item(), 4))
    print("  std   :", round(desc_emb.std().item(),  4))
    nz_desc = (desc_emb.abs().sum(dim=1) > 0).sum().item()
    print(f"  nonzero rows: {nz_desc} / {N}")

    print("\n===== TWEET EMB DEBUG =====")
    print("  shape :", tuple(tweet_emb.shape))
    sample = tweet_emb[:100]
    print("  sample mean:", round(sample.mean().item(), 4))
    print("  sample std :", round(sample.std().item(), 4))
    users_with_tw_emb = (tweet_len.sum(dim=1) > 0).sum().item()
    print(f"  users with tweets: {users_with_tw_emb} / {N}")
    total_tok = N * MAX_TWEETS_PER_USER * MAX_WORDS_PER_TWEET
    active_tok = int(tweet_len.sum().item())
    print(f"  active token lengths: {active_tok} / {total_tok}")

    print("\n===== TOKEN LENGTH CHECK =====")
    for i in range(min(3, N)):
        print(f"  user {i} tweet lengths: {tweet_len[i].tolist()}")

    print("\n===== RAW TEXT VS EMB CHECK =====")
    sample_uid = user_ids[0]
    print(f"  user: {sample_uid}")
    print(f"  tweets: {tweet_texts.get(sample_uid, [])[:2]}")
    print("\n[Stage 4] Building heterogeneous CACL edges …")
    hetero_edges, tweet_to_idx, list_to_idx, hashtag_to_idx = build_hetero_edges(
        EDGE_CSV_PATH, user_ids
    )
    # ── Save ──────────────────────────────────────────────────────────────
    out = {
    "user_ids": user_ids,
    "labels": labels,
    "split": split,

    # For BotDCGC
    "edge_index": edge_index,

    # For shared encoder
    "num": num,
    "cat": cat,
    "desc_emb": desc_emb,
    "tweet_emb": tweet_emb,
    "tweet_len": tweet_len,

    # For CACL heterogeneous graph
    "hetero_edges": hetero_edges,
    "tweet_to_idx": tweet_to_idx,
    "list_to_idx": list_to_idx,
    "hashtag_to_idx": hashtag_to_idx,
    }

    print("\nModel-ready shapes:")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k:12s}: {tuple(v.shape)}")

    print(f"\nSaving → {OUTPUT_PATH}")
    torch.save(out, OUTPUT_PATH)
    size_gb = OUTPUT_PATH.stat().st_size / 1e9
    print(f"Saved ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()