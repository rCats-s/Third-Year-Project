from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


INPUT_PATH = Path(r"D:\3rd year project\Implementation\Datasets\twibot22_out\graph_data.pt")
OUTPUT_PATH = Path(r"D:\3rd year project\Implementation\Datasets\twibot22_out\graph_data_model_ready_real.pt")

USER_JSON_PATH = INPUT_PATH.parent / "user.json"
TWEET_JSON_PATH = INPUT_PATH.parent / "tweet_0.json"

MODEL_NAME = "roberta-base"
ROBERTA_DIM = 768
MAX_TWEETS_PER_USER = 20
MAX_WORDS_PER_TWEET = 32
MAX_DESC_TOKENS = 64


def normalise_user_id(x) -> str:
    x = str(x)
    return x if x.startswith("u") else f"u{x}"


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def zscore_clip(x: np.ndarray, clip_value: float = 3.0) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    z = (x - mu) / std
    return np.clip(z, -clip_value, clip_value)


def rebuild_timestamps_from_tweet_json(tweet_json_path: Path, user_ids: List[str]) -> Dict[str, List[int]]:
    wanted = set(user_ids)
    out = defaultdict(list)

    with open(tweet_json_path, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    for tw in tweets:
        author = normalise_user_id(
            tw.get("author_id", tw.get("user", {}).get("id", ""))
        )
        if author not in wanted:
            continue

        created_at = tw.get("created_at")
        if not created_at:
            continue

        try:
            dt = datetime.fromisoformat(created_at)
            out[author].append(int(dt.timestamp()))
        except Exception:
            continue

    return dict(out)


def posting_hour_entropy(timestamps: List[int]) -> float:
    valid = [t for t in timestamps if isinstance(t, (int, float)) and t > 0]
    if not valid:
        return 0.0

    hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in valid]
    counts = np.bincount(hours, minlength=24).astype(np.float32)
    probs = counts / max(counts.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def compute_temporal_features(tweet_timestamps: Dict[str, List[int]], user_ids: List[str]):
    temporal_num = np.zeros((len(user_ids), 6), dtype=np.float32)
    temporal_cat = np.zeros((len(user_ids), 4), dtype=np.float32)

    for i, uid in enumerate(user_ids):
        raw_ts = tweet_timestamps.get(uid, [])
        ts = sorted(int(t) for t in raw_ts if isinstance(t, (int, float)) and int(t) > 0)

        if len(ts) == 0:
            continue

        if len(ts) >= 2:
            gaps = np.diff(ts).astype(np.float32) / 3600.0
            mean_gap = float(gaps.mean())
            std_gap = float(gaps.std())
        else:
            gaps = np.array([], dtype=np.float32)
            mean_gap = 0.0
            std_gap = 0.0

        days = [int(t // 86400) for t in ts]
        unique_days = len(set(days))
        tweets_per_active_day = safe_ratio(len(ts), unique_days)

        entropy = posting_hour_entropy(ts)

        weekdays = [datetime.fromtimestamp(t, tz=timezone.utc).weekday() for t in ts]
        weekend_count = sum(1 for w in weekdays if w >= 5)
        weekend_ratio = safe_ratio(weekend_count, len(ts))

        hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts]
        night_count = sum(1 for h in hours if h < 6 or h >= 23)
        night_ratio = safe_ratio(night_count, len(ts))

        temporal_num[i] = np.array(
            [mean_gap, std_gap, tweets_per_active_day, entropy, weekend_ratio, night_ratio],
            dtype=np.float32
        )

        cv = safe_ratio(std_gap, mean_gap) if mean_gap > 0 else 0.0
        temporal_cat[i, 0] = 1.0 if night_ratio >= 0.50 else 0.0
        temporal_cat[i, 1] = 1.0 if weekend_ratio >= 0.50 else 0.0
        temporal_cat[i, 2] = 1.0 if (len(gaps) >= 3 and cv < 0.50) else 0.0
        temporal_cat[i, 3] = 1.0 if (len(gaps) >= 3 and cv > 1.50) else 0.0

    return zscore_clip(temporal_num), temporal_cat


def load_user_descriptions(user_json_path: Path) -> Dict[str, str]:
    with open(user_json_path, "r", encoding="utf-8") as f:
        users = json.load(f)

    out = {}
    for u in users:
        uid = normalise_user_id(u.get("id", ""))
        desc = u.get("description") or ""
        out[uid] = str(desc)

    return out


def load_user_tweets(tweet_json_path: Path, user_ids: List[str], max_tweets_per_user: int) -> Dict[str, List[str]]:
    wanted = set(user_ids)
    out = defaultdict(list)

    with open(tweet_json_path, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    for tw in tweets:
        author = normalise_user_id(
            tw.get("author_id", tw.get("user", {}).get("id", ""))
        )
        if author not in wanted:
            continue
        if len(out[author]) >= max_tweets_per_user:
            continue

        text = tw.get("full_text", tw.get("text", ""))
        out[author].append(str(text))

    return dict(out)


@torch.no_grad()
def build_desc_emb(user_ids: List[str], desc_lookup: Dict[str, str], tokenizer, model, device) -> torch.Tensor:
    desc_emb = torch.zeros((len(user_ids), ROBERTA_DIM), dtype=torch.float32)

    model.eval()
    for i, uid in enumerate(user_ids):
        text = desc_lookup.get(uid, "")
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_DESC_TOKENS,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        # RoBERTa has no pooled_output; use first token representation
        desc_emb[i] = outputs.last_hidden_state[0, 0].detach().cpu()

    return desc_emb


@torch.no_grad()
def build_tweet_emb_and_len(
    user_ids: List[str],
    tweet_lookup: Dict[str, List[str]],
    tokenizer,
    model,
    device
) -> tuple[torch.Tensor, torch.Tensor]:
    tweet_emb = torch.zeros(
        (len(user_ids), MAX_TWEETS_PER_USER, MAX_WORDS_PER_TWEET, ROBERTA_DIM),
        dtype=torch.float32
    )
    tweet_len = torch.zeros((len(user_ids), MAX_TWEETS_PER_USER), dtype=torch.long)

    model.eval()
    for i, uid in enumerate(user_ids):
        tweets = tweet_lookup.get(uid, [])[:MAX_TWEETS_PER_USER]

        for j, text in enumerate(tweets):
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_WORDS_PER_TWEET,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)

            attn = encoded["attention_mask"][0]
            valid_len = int(attn.sum().item())
            tweet_len[i, j] = min(valid_len, MAX_WORDS_PER_TWEET)

            hidden = outputs.last_hidden_state[0].detach().cpu()  # (max_words, 768)
            tweet_emb[i, j] = hidden[:MAX_WORDS_PER_TWEET]

    return tweet_emb, tweet_len


def main():
    print(f"Loading intermediate graph: {INPUT_PATH}")
    data = torch.load(INPUT_PATH)

    user_ids = data["user_ids"]
    labels = data["labels"]
    split = data["split"]
    edge_index = data["edge_index"]

    user_num_feat = data["user_num_feat"].cpu().numpy()
    user_cat_feat = data["user_cat_feat"].cpu().numpy()

    tweet_timestamps = rebuild_timestamps_from_tweet_json(TWEET_JSON_PATH, user_ids)
    nonzero_users = sum(1 for xs in tweet_timestamps.values() if any(t > 0 for t in xs))
    print("users with valid rebuilt timestamps:", nonzero_users)
    assert nonzero_users > 0, "No valid timestamps rebuilt"

    temporal_num, temporal_cat = compute_temporal_features(tweet_timestamps, user_ids)
    num = torch.tensor(np.concatenate([user_num_feat, temporal_num], axis=1), dtype=torch.float32)
    cat = torch.tensor(np.concatenate([user_cat_feat, temporal_cat], axis=1), dtype=torch.float32)

    desc_lookup = load_user_descriptions(USER_JSON_PATH)
    tweet_lookup = load_user_tweets(TWEET_JSON_PATH, user_ids, MAX_TWEETS_PER_USER)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    desc_emb = build_desc_emb(user_ids, desc_lookup, tokenizer, model, device)
    print("\n===== DESC EMB DEBUG =====")
    print("desc_emb shape:", tuple(desc_emb.shape))
    print("desc_emb mean:", desc_emb.mean().item())
    print("desc_emb std :", desc_emb.std().item())
    print("desc_emb min/max:", desc_emb.min().item(), desc_emb.max().item())

    # check non-zero rows
    nonzero_desc = (desc_emb.abs().sum(dim=1) > 0).sum().item()
    print("nonzero desc embeddings:", nonzero_desc, "/", desc_emb.shape[0])

    # sample vector
    print("sample desc_emb[0][:10]:", desc_emb[0][:10])
    tweet_emb, tweet_len = build_tweet_emb_and_len(user_ids, tweet_lookup, tokenizer, model, device)
    print("\n===== TWEET EMB DEBUG =====")
    print("tweet_emb shape:", tuple(tweet_emb.shape))
    print("tweet_len shape:", tuple(tweet_len.shape))

    print("tweet_emb mean:", tweet_emb.mean().item())
    print("tweet_emb std :", tweet_emb.std().item())

    # how many users actually have tweets
    users_with_tweets = (tweet_len.sum(dim=1) > 0).sum().item()
    print("users with tweets:", users_with_tweets, "/", tweet_len.shape[0])

    # sample user
    sample_user = 0
    print("\nSample user tweet lengths:", tweet_len[sample_user])

    # count non-zero tokens
    nonzero_tokens = (tweet_emb.abs().sum(dim=-1) > 0).sum().item()
    total_tokens = tweet_emb.shape[0] * tweet_emb.shape[1] * tweet_emb.shape[2]
    print("nonzero token embeddings:", nonzero_tokens, "/", total_tokens)

    # inspect first tweet embedding
    print("\nsample tweet_emb[0,0,0][:10]:", tweet_emb[0,0,0][:10])

    out = {
        "user_ids": user_ids,
        "labels": labels,
        "split": split,
        "edge_index": edge_index,
        "num": num,
        "cat": cat,
        "desc_emb": desc_emb,
        "tweet_emb": tweet_emb,
        "tweet_len": tweet_len,
    }

    print("Model-ready shapes:")
    print("num      :", tuple(num.shape))
    print("cat      :", tuple(cat.shape))
    print("desc_emb :", tuple(desc_emb.shape))
    print("tweet_emb:", tuple(tweet_emb.shape))
    print("tweet_len:", tuple(tweet_len.shape))
    print("edge_index:", tuple(edge_index.shape))

    torch.save(out, OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")

    print("\n===== TOKEN LENGTH CHECK =====")
    for i in range(3):
        print(f"user {i} tweet lengths:", tweet_len[i].tolist())
    print("\n===== RAW TEXT VS EMB CHECK =====")
    sample_uid = user_ids[0]
    print("user:", sample_uid)
    print("tweets:", tweet_lookup.get(sample_uid, [])[:2])

if __name__ == "__main__":
    main()