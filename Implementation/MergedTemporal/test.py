from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch


INPUT_PATH = Path(r"D:\3rd year project\Implementation\Datasets\twibot22_smoke\graph_data.pt")
OUTPUT_PATH = Path(r"D:\3rd year project\Implementation\Datasets\twibot22_smoke\graph_data_model_ready.pt")

# For smoke testing only
ROBERTA_DIM = 768
MAX_TWEETS_PER_USER = 5
MAX_WORDS_PER_TWEET = 8

def normalise_user_id(x) -> str:
    x = str(x)
    return x if x.startswith("u") else f"u{x}"


def rebuild_timestamps_from_tweet_json(tweet_json_path, user_ids):
    """
    Rebuild per-user timestamps directly from smoke subset tweet_0.json.

    Returns:
        dict[str, list[int]]
    """
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

        created_at = tw.get("created_at", None)
        if not created_at:
            continue

        try:
            dt = datetime.fromisoformat(created_at)
            ts = int(dt.timestamp())
            out[author].append(ts)
        except Exception:
            continue

    return dict(out)

def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def zscore_clip(x: np.ndarray, clip_value: float = 3.0) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    z = (x - mu) / std
    return np.clip(z, -clip_value, clip_value)


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
    """
    Returns:
        temporal_num: (N, 6)
        temporal_cat: (N, 4)

    Numerical features:
        0 mean inter-tweet gap (hours)
        1 std inter-tweet gap (hours)
        2 tweets per active day
        3 posting-hour entropy
        4 weekend posting ratio
        5 night posting ratio

    Categorical features:
        0 night-heavy account
        1 weekend-heavy account
        2 highly regular posting
        3 bursty posting
    """
    temporal_num = np.zeros((len(user_ids), 6), dtype=np.float32)
    temporal_cat = np.zeros((len(user_ids), 4), dtype=np.float32)

    for i, uid in enumerate(user_ids):
        raw_ts = tweet_timestamps.get(uid, [])
        ts = sorted(int(t) for t in raw_ts if isinstance(t, (int, float)) and int(t) > 0)

        if len(ts) == 0:
            continue

        if len(ts) >= 2:
            gaps = np.diff(ts).astype(np.float32) / 3600.0  # hours
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

        temporal_num[i] = np.array([
            mean_gap,
            std_gap,
            tweets_per_active_day,
            entropy,
            weekend_ratio,
            night_ratio,
        ], dtype=np.float32)

        # Simple thresholded categorical flags
        cv = safe_ratio(std_gap, mean_gap) if mean_gap > 0 else 0.0
        temporal_cat[i, 0] = 1.0 if night_ratio >= 0.50 else 0.0
        temporal_cat[i, 1] = 1.0 if weekend_ratio >= 0.50 else 0.0
        temporal_cat[i, 2] = 1.0 if (len(gaps) >= 3 and cv < 0.50) else 0.0   # regular
        temporal_cat[i, 3] = 1.0 if (len(gaps) >= 3 and cv > 1.50) else 0.0   # bursty

    temporal_num = zscore_clip(temporal_num, clip_value=3.0)
    return temporal_num, temporal_cat


def build_placeholder_desc_emb(num_users: int, dim: int = ROBERTA_DIM) -> torch.Tensor:
    # Shape sanity only; replace later with real description embeddings
    return torch.zeros((num_users, dim), dtype=torch.float32)


def text_to_fake_token_embeddings(
    texts_per_user: Dict[str, List[str]],
    user_ids: List[str],
    max_tweets: int = MAX_TWEETS_PER_USER,
    max_words: int = MAX_WORDS_PER_TWEET,
    roberta_dim: int = ROBERTA_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Smoke-test placeholder tweet embeddings.
    Produces shapes expected by TweetEncoder:
        tweet_emb: (N, M, max_words, roberta_dim)
        tweet_len: (N, M)

    This does NOT create real RoBERTa embeddings.
    It just fills deterministic pseudo-embeddings from token hashes so the
    pipeline can be sanity-checked end-to-end.
    """
    n = len(user_ids)
    tweet_emb = torch.zeros((n, max_tweets, max_words, roberta_dim), dtype=torch.float32)
    tweet_len = torch.zeros((n, max_tweets), dtype=torch.long)

    for i, uid in enumerate(user_ids):
        tweets = texts_per_user.get(uid, [])[:max_tweets]

        for j, text in enumerate(tweets):
            tokens = str(text).split()[:max_words]
            tweet_len[i, j] = len(tokens)

            for k, tok in enumerate(tokens):
                # deterministic tiny pseudo-vector based on token hash
                seed = abs(hash(tok)) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(roberta_dim).astype(np.float32) * 0.02
                tweet_emb[i, j, k] = torch.from_numpy(vec)

    return tweet_emb, tweet_len
def inspect_one_user_raw_vs_rebuilt(tweet_json_path, target_uid, rebuilt_map, max_items=5):
    with open(tweet_json_path, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    raw_vals = []
    for tw in tweets:
        author = normalise_user_id(
            tw.get("author_id", tw.get("user", {}).get("id", ""))
        )
        if author == target_uid:
            raw_vals.append(tw.get("created_at"))

    print(f"\nRaw created_at values for {target_uid}:")
    for x in raw_vals[:max_items]:
        print(x)

    print(f"\nRebuilt timestamps for {target_uid}:")
    for ts in rebuilt_map.get(target_uid, [])[:max_items]:
        print(ts, "->", datetime.fromtimestamp(ts, tz=timezone.utc).isoformat())
def validate_timestamp_rebuild(tweet_json_path, rebuilt_map, max_checks=20):
    with open(tweet_json_path, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    checked = 0
    for tw in tweets:
        author = normalise_user_id(
            tw.get("author_id", tw.get("user", {}).get("id", ""))
        )
        created_at = tw.get("created_at")
        if not created_at or author not in rebuilt_map:
            continue

        try:
            expected = int(datetime.fromisoformat(created_at).timestamp())
        except Exception:
            continue

        if expected not in rebuilt_map[author]:
            raise AssertionError(
                f"Timestamp mismatch for {author}: raw={created_at}, expected={expected}"
            )

        checked += 1
        if checked >= max_checks:
            break

    print(f"validated {checked} raw->rebuilt timestamps successfully")
def main():
    print(f"Loading: {INPUT_PATH}")
    data = torch.load(INPUT_PATH)

    user_ids = data["user_ids"]
    labels = data["labels"]
    split = data["split"]
    edge_index = data["edge_index"]

    user_num_feat = data["user_num_feat"].cpu().numpy()   # (N, 7)
    user_cat_feat = data["user_cat_feat"].cpu().numpy()   # (N, 11)
    tweet_texts = data["tweet_texts"]
    tweet_json_path = INPUT_PATH.parent / "tweet_0.json"
    tweet_timestamps = rebuild_timestamps_from_tweet_json(tweet_json_path, user_ids)

    nonzero_users = sum(
        1 for ts_list in tweet_timestamps.values()
        if any(t > 0 for t in ts_list)
    )
    print("users with at least one valid rebuilt timestamp:", nonzero_users)

    sample_uid = next(iter(tweet_timestamps), None)
    if sample_uid is not None:
        print("sample rebuilt datetimes:")
        inspect_one_user_raw_vs_rebuilt(tweet_json_path, sample_uid, tweet_timestamps)
        for ts in tweet_timestamps[sample_uid][:10]:
            print(datetime.fromtimestamp(ts, tz=timezone.utc).isoformat())
    assert nonzero_users > 0, "No valid timestamps rebuilt from tweet_0.json"
    validate_timestamp_rebuild(tweet_json_path, tweet_timestamps, max_checks=20)
    
    print("Base keys:", list(data.keys()))
    print("user_num_feat:", user_num_feat.shape)
    print("user_cat_feat:", user_cat_feat.shape)
    print("edge_index:", tuple(edge_index.shape))
    print("tweet_text users:", len(tweet_texts))
    print("tweet_timestamp users:", len(tweet_timestamps))

    temporal_num, temporal_cat = compute_temporal_features(tweet_timestamps, user_ids)

    num = np.concatenate([user_num_feat, temporal_num], axis=1)   # (N, 13)
    cat = np.concatenate([user_cat_feat, temporal_cat], axis=1)   # (N, 15)

    num = torch.tensor(num, dtype=torch.float32)
    cat = torch.tensor(cat, dtype=torch.float32)

    desc_emb = build_placeholder_desc_emb(len(user_ids), dim=ROBERTA_DIM)
    tweet_emb, tweet_len = text_to_fake_token_embeddings(
        tweet_texts,
        user_ids,
        max_tweets=MAX_TWEETS_PER_USER,
        max_words=MAX_WORDS_PER_TWEET,
        roberta_dim=ROBERTA_DIM,
    )

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
        "temporal_num_fields": [
            "mean_intertweet_gap_hours",
            "std_intertweet_gap_hours",
            "tweets_per_active_day",
            "posting_hour_entropy",
            "weekend_post_ratio",
            "night_post_ratio",
        ],
        "temporal_cat_fields": [
            "night_heavy",
            "weekend_heavy",
            "highly_regular",
            "bursty",
        ],
    }

    print("\nModel-ready shapes:")
    print("num      :", tuple(out["num"].shape))
    print("cat      :", tuple(out["cat"].shape))
    print("desc_emb :", tuple(out["desc_emb"].shape))
    print("tweet_emb:", tuple(out["tweet_emb"].shape))
    print("tweet_len:", tuple(out["tweet_len"].shape))
    print("labels   :", tuple(out["labels"].shape))
    print("edge_index:", tuple(out["edge_index"].shape))

    assert out["num"].shape[1] == 13, f"Expected 13 num features, got {out['num'].shape[1]}"
    assert out["cat"].shape[1] == 15, f"Expected 15 cat features, got {out['cat'].shape[1]}"
    assert not torch.isnan(out["num"]).any(), "num contains NaN"
    assert not torch.isnan(out["cat"]).any(), "cat contains NaN"
    assert not torch.isnan(out["desc_emb"]).any(), "desc_emb contains NaN"
    assert not torch.isnan(out["tweet_emb"]).any(), "tweet_emb contains NaN"

    torch.save(out, OUTPUT_PATH)
    print(f"\nSaved model-ready graph to:\n{OUTPUT_PATH}")
    all_valid = [t for ts_list in tweet_timestamps.values() for t in ts_list if t > 0]

    print("min timestamp:", min(all_valid) if all_valid else None)
    print("max timestamp:", max(all_valid) if all_valid else None)

    if all_valid:
        print("min datetime:", datetime.fromtimestamp(min(all_valid), tz=timezone.utc).isoformat())
        print("max datetime:", datetime.fromtimestamp(max(all_valid), tz=timezone.utc).isoformat())


if __name__ == "__main__":
    main()