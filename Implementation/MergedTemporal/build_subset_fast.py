"""
build_subset_fast.py
====================
Optimised TwiBot-22 subsetter for low-resource hardware (HDD, 8 GB RAM).

Key differences from the original script
-----------------------------------------
• Stage 2 is fully vectorised — no iterrows(), no per-row Python calls.
  A 6 GB edge file that took 1.5-2 h now takes roughly 5-10 minutes.
• Stage 3 has early exit: scanning stops as soon as every sampled user's
  tweet budget is satisfied, rather than always reading the full file.
• Two modes controlled by --mode:

    graph   (default / fast)
            Skips all tweet files entirely.
            Outputs user.json, edge.csv, label.csv, split.csv, list.json,
            hashtag.json and a ready-to-load graph_data.pt.
            End-to-end runtime on HDD: ~10-20 minutes for 10 k users.

    full    Adds tweet streaming on top of graph mode.
            Runtime depends on --max_tweet_files and how densely the
            sampled users' tweets appear near the start of each file.

Verification / smoke-test
--------------------------
    python build_subset_fast.py \
        --input_dir  D:/twibot22 \
        --output_dir D:/subset_verify \
        --n_users    300           \
        --mode       graph

    Completes in ~2 minutes and lets you validate your dataloader end-to-end
    before committing to a full overnight run.

Full experiment run (graph-only, no tweets)
--------------------------------------------
    python build_subset_fast.py \
        --input_dir  D:/twibot22 \
        --output_dir D:/subset_10k \
        --n_users    10000

Full run with tweets (one file scan)
-------------------------------------
    python build_subset_fast.py \
        --input_dir  D:/twibot22 \
        --output_dir D:/subset_10k_tweets \
        --n_users    10000 \
        --mode       full \
        --max_tweets_per_user 15 \
        --max_tweet_files 1

Requirements
------------
    pip install ijson pandas tqdm torch
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from decimal import Decimal
import math

try:
    import ijson
except ImportError:
    sys.exit("[ERROR] Run:  pip install ijson")


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# Relations whose source is a user and target is another user
USER_USER      = frozenset({"following", "followers", "followed"})

# Relations whose source is a user and target is a tweet
USER_TO_TWEET  = frozenset({"pinned", "post", "retweeted", "like",
                             "quoted", "replied", "mentioned"})

# Relations whose source is a user and target is a list
USER_TO_LIST   = frozenset({"own", "membership"})

# Relations whose source is a user and target is a hashtag
USER_TO_HASHTAG = frozenset({"discuss", "contain"})

ALL_RELATIONS = USER_USER | USER_TO_TWEET | USER_TO_LIST | USER_TO_HASHTAG


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════
def normalise_user_id(x) -> str:
    x = str(x)
    return x if x.startswith("u") else f"u{x}"

def _timer(label: str):
    """Simple context manager that prints elapsed time."""
    class _T:
        def __enter__(self):
            self._t = time.time()
            return self
        def __exit__(self, *_):
            print(f"    [{label}] {time.time()-self._t:.1f}s")
    return _T()


def normalise_edge_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df.rename(columns={
        "source_id": "source", "target_id": "target",
        "src": "source", "tgt": "target",
        "relation_type": "relation",
    })


def normalise_meta_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df.rename(columns={"uid": "id"})


def record_id(record: dict) -> str:
    return str(record.get("id", record.get("id_str", record.get("tag", ""))))


def load_json_file(path: Path, desc: str) -> list:
    print(f"  loading {desc} ({path.stat().st_size/1e6:.0f} MB)…", end=" ", flush=True)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"{len(data):,} records")
    return data


def _json_default(obj):
    if isinstance(obj, Decimal):
        # Preserve integers cleanly
        if obj == obj.to_integral_value():
            return int(obj)
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def write_json_file(data: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=_json_default)
    print(f"  wrote {path.name}  ({len(data):,} records, {path.stat().st_size/1e6:.1f} MB)")

def filter_json_array_stream(path: Path, wanted_ids: set, desc: str) -> list:
    print(f"  streaming {desc} ({path.stat().st_size/1e6:.0f} MB)…", flush=True)
    kept = []

    with open(path, "rb") as f:
        items = ijson.items(f, "item")
        for obj in items:
            if record_id(obj) in wanted_ids:
                kept.append(obj)

    print(f"  → {len(kept):,} {desc} kept")
    return kept

# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 – stratified user sampling
# ═══════════════════════════════════════════════════════════════════════════

def normalise_user_id(x) -> str:
    x = str(x)
    return x if x.startswith("u") else f"u{x}"


def stage1_sample_users(
    input_dir: Path,
    n_users:   int,
    seed:      int,
) -> tuple[set, pd.DataFrame, pd.DataFrame]:
    """
    Stratified sample across (split × label) groups.
    Returns (sampled_user_id_set, label_df, split_df).
    """
    print("\n[Stage 1]  Stratified user sampling")

    label_df = normalise_meta_df(pd.read_csv(input_dir / "label.csv", dtype=str))
    split_df = normalise_meta_df(pd.read_csv(input_dir / "split.csv", dtype=str))

    # NORMALISE USER IDS
    label_df["id"] = label_df["id"].astype(str).map(normalise_user_id)
    split_df["id"] = split_df["id"].astype(str).map(normalise_user_id)

    label_df["label"] = label_df["label"].str.lower()
    split_df["split"] = split_df["split"].str.lower().replace({"dev": "valid", "val": "valid"})

    merged = label_df.merge(split_df, on="id", how="inner")
    total  = len(merged)

    if n_users > total:
        print(f"  requested {n_users:,} but only {total:,} users available — using all")
        n_users = total

    rng = random.Random(seed)
    parts = []

    for (split, label), grp in merged.groupby(["split", "label"]):
        target = max(1, round(n_users * len(grp) / total))
        target = min(target, len(grp))
        sampled_ids = rng.sample(grp["id"].tolist(), target)
        parts.append(grp[grp["id"].isin(sampled_ids)])
        print(f"  {split:>5} | {label:>5} | avail={len(grp):>7,} | sampled={target:>6,}")

    selected = pd.concat(parts, ignore_index=True)
    sampled_users = set(selected["id"])

    print(f"\n  total sampled users: {len(sampled_users):,}")
    return sampled_users, label_df, split_df


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 – vectorised edge scan  (the main fix)
# ═══════════════════════════════════════════════════════════════════════════

def stage2_scan_edges(
    input_dir: Path,
    sampled_users: set,
) -> tuple[pd.DataFrame, set, set, set]:
    """
    Read edge CSV in chunks using vectorised pandas masks.

    Important:
    - user IDs are normalised to the TwiBot-22 user format: u<id>
    - tweet/list/hashtag target IDs are left unchanged
    """
    print("\n[Stage 2]  Vectorised edge scan")

    edge_files = sorted(input_dir.glob("edge*.csv"))
    if not edge_files:
        sys.exit("[ERROR] No edge CSV found.")
    edge_path = edge_files[0]
    file_mb = edge_path.stat().st_size / 1e6
    print(f"  file: {edge_path.name}  ({file_mb:.0f} MB)")

    tweet_ids: Set[str] = set()
    list_ids: Set[str] = set()
    hashtag_ids: Set[str] = set()
    kept_chunks: List[pd.DataFrame] = []

    chunk_size = 500_000

    with tqdm(
        total=int(file_mb * 1e6 // 130),
        unit="rows",
        unit_scale=True,
        desc="  scanning edge CSV",
    ) as bar:

        for chunk in pd.read_csv(
            edge_path,
            chunksize=chunk_size,
            dtype=str,
            low_memory=False,
        ):
            chunk = normalise_edge_df(chunk)
            chunk["source"] = chunk["source"].astype(str)
            chunk["target"] = chunk["target"].astype(str)
            chunk["relation"] = chunk["relation"].astype(str)

            n = len(chunk)
            bar.update(n)

            rel = chunk["relation"]

            # source is always treated as a user ID for the relations we care about
            src_user = chunk["source"].map(normalise_user_id)
            src_ok = src_user.isin(sampled_users)

            # target is only a user ID for USER_USER relations
            tgt_user = chunk["target"].map(normalise_user_id)
            tgt_ok = tgt_user.isin(sampled_users)

            uu_mask = rel.isin(USER_USER) & src_ok & tgt_ok
            ut_mask = rel.isin(USER_TO_TWEET) & src_ok
            ul_mask = rel.isin(USER_TO_LIST) & src_ok
            uh_mask = rel.isin(USER_TO_HASHTAG) & src_ok

            # Build kept chunk with normalized user IDs where appropriate
            kept_parts = []

            if uu_mask.any():
                uu_chunk = pd.DataFrame({
                    "source": src_user[uu_mask].values,
                    "target": tgt_user[uu_mask].values,
                    "relation": rel[uu_mask].values,
                })
                kept_parts.append(uu_chunk)

            if ut_mask.any():
                ut_chunk = pd.DataFrame({
                    "source": src_user[ut_mask].values,
                    "target": chunk.loc[ut_mask, "target"].values,
                    "relation": rel[ut_mask].values,
                })
                kept_parts.append(ut_chunk)

            if ul_mask.any():
                ul_chunk = pd.DataFrame({
                    "source": src_user[ul_mask].values,
                    "target": chunk.loc[ul_mask, "target"].values,
                    "relation": rel[ul_mask].values,
                })
                kept_parts.append(ul_chunk)

            if uh_mask.any():
                uh_chunk = pd.DataFrame({
                    "source": src_user[uh_mask].values,
                    "target": chunk.loc[uh_mask, "target"].values,
                    "relation": rel[uh_mask].values,
                })
                kept_parts.append(uh_chunk)

            if kept_parts:
                kept_chunks.append(pd.concat(kept_parts, ignore_index=True))

            # collect entity IDs exactly as stored in edge.csv
            tweet_ids.update(chunk.loc[ut_mask, "target"].tolist())
            list_ids.update(chunk.loc[ul_mask, "target"].tolist())
            hashtag_ids.update(chunk.loc[uh_mask, "target"].tolist())

    if kept_chunks:
        all_edges = pd.concat(kept_chunks, ignore_index=True)
    else:
        all_edges = pd.DataFrame(columns=["source", "target", "relation"])

    print(
        f"  edges kept: {len(all_edges):,}  |  "
        f"tweet IDs: {len(tweet_ids):,}  |  "
        f"list IDs: {len(list_ids):,}  |  "
        f"hashtag IDs: {len(hashtag_ids):,}"
    )
    return all_edges, tweet_ids, list_ids, hashtag_ids


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 – tweet streaming with early exit
# ═══════════════════════════════════════════════════════════════════════════

def _stream_one_tweet_file(path, sampled_users, budget, users_left) -> list:
    kept = []
    fsize = path.stat().st_size

    with open(path, "rb") as f, tqdm(
        total=fsize, unit="B", unit_scale=True,
        desc=f"    {path.name}", leave=False,
    ) as bar:
        prev_pos = 0
        items = ijson.items(f, "item")

        for tweet in items:
            if users_left[0] is not None and users_left[0] == 0:
                bar.update(fsize - prev_pos)
                break

            try:
                pos = f.tell()
                bar.update(pos - prev_pos)
                prev_pos = pos
            except Exception:
                pass

            author = normalise_user_id(tweet.get("author_id", tweet.get("user", {}).get("id", "")))
            if author not in sampled_users:
                continue
            if budget.get(author, 0) <= 0:
                continue

            kept.append(tweet)
            budget[author] -= 1

            if users_left[0] is not None and budget[author] == 0:
                users_left[0] -= 1

    return kept


def stage3_stream_tweets(
    input_dir: Path,
    tweet_ids: set,
    sampled_users: set,
    max_tweets_per_user: int,
    max_tweet_files: int,
) -> list:
    print("\n[Stage 3]  Streaming tweet files (with early exit)")

    tweet_files = sorted(input_dir.glob("tweet_*.json"))
    if max_tweet_files > 0:
        tweet_files = tweet_files[:max_tweet_files]

    if not tweet_files:
        print("  [WARN] No tweet_*.json files found — skipping.")
        return []

    print(f"  files to scan: {len(tweet_files)}  "
          f"(of {len(sorted(input_dir.glob('tweet_*.json')))} total)")

    no_cap = max_tweets_per_user < 0

    if no_cap:
        budget = {uid: float("inf") for uid in sampled_users}
        users_left = [None]
    else:
        budget = {uid: max_tweets_per_user for uid in sampled_users}
        users_left = [len(sampled_users)]

    all_tweets = []
    for tfile in tweet_files:
        if users_left[0] is not None and users_left[0] == 0:
            print("  all user budgets satisfied — skipping remaining files")
            break

        chunk = _stream_one_tweet_file(tfile, sampled_users, budget, users_left)
        all_tweets.extend(chunk)

        if users_left[0] is None:
            print(f"  {tfile.name}: kept {len(chunk):,}  "
                  f"(total: {len(all_tweets):,}, no cap mode)")
        else:
            print(f"  {tfile.name}: kept {len(chunk):,}  "
                  f"(total: {len(all_tweets):,}, users still needing tweets: {users_left[0]})")

    return all_tweets


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 4 – filter user / list / hashtag JSON
# ═══════════════════════════════════════════════════════════════════════════

def stage4_filter_entity_json(input_dir: Path, sampled_users: set, list_ids: set, hashtag_ids: set):
    print("\n[Stage 4]  Filtering entity JSON files")

    sub_users = filter_json_array_stream(input_dir / "user.json", sampled_users, "users")

    sub_lists = []
    list_path = input_dir / "list.json"
    if list_path.exists():
        sub_lists = filter_json_array_stream(list_path, list_ids, "lists")

    sub_hashtags = []
    hashtag_path = input_dir / "hashtag.json"
    if hashtag_path.exists():
        sub_hashtags = filter_json_array_stream(hashtag_path, hashtag_ids, "hashtags")

    return sub_users, sub_lists, sub_hashtags


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 5 – write raw schema files  (TwiBot-22 compatible)
# ═══════════════════════════════════════════════════════════════════════════

def stage5_write_raw_schema(
    output_dir:   Path,
    sub_users:    list,
    all_tweets:   list,
    sub_lists:    list,
    sub_hashtags: list,
    all_edges:    pd.DataFrame,
    label_df:     pd.DataFrame,
    split_df:     pd.DataFrame,
    sampled_users: set,
    max_tweets_per_user: int,
):
    print("\n[Stage 5]  Writing raw-schema subset files")
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json_file(sub_users,    output_dir / "user.json")
    write_json_file(all_tweets,   output_dir / "tweet_0.json")
    write_json_file(sub_lists,    output_dir / "list.json")
    write_json_file(sub_hashtags, output_dir / "hashtag.json")

    edge_out = output_dir / "edge.csv"
    all_edges.rename(columns={
        "source": "source_id", "target": "target_id", "relation": "relation"
    }).to_csv(edge_out, index=False)
    print(f"  wrote edge.csv  ({len(all_edges):,} rows, {edge_out.stat().st_size/1e6:.1f} MB)")

    label_sub = label_df[label_df["id"].isin(sampled_users)].copy()
    label_sub.to_csv(output_dir / "label.csv", index=False)
    print(f"  wrote label.csv  ({len(label_sub):,} rows)")

    split_sub = split_df[split_df["id"].isin(sampled_users)].copy()
    split_sub.to_csv(output_dir / "split.csv", index=False)
    print(f"  wrote split.csv  ({len(split_sub):,} rows)")


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 6 – write graph_data.pt  (ML pipeline ready)
# ═══════════════════════════════════════════════════════════════════════════

def stage6_write_graph_tensors(
    output_dir:    Path,
    sub_users:     list,
    all_edges:     pd.DataFrame,
    label_df:      pd.DataFrame,
    split_df:      pd.DataFrame,
    sampled_users: set,
    all_tweets:    list,
):
    """
    Build and save graph_data.pt — a dict of tensors ready for your dataloader.

    Contents
    --------
    user_ids        : list[str]            ordered user IDs (index = node index)
    labels          : LongTensor (N,)      0=human, 1=bot, -1=unlabelled
    split           : LongTensor (N,)      0=train, 1=valid, 2=test
    edge_index      : LongTensor (2, E)    user-user following edges
    user_num_feat   : FloatTensor (N, F_n) numerical user features (z-scored)
    user_cat_feat   : FloatTensor (N, F_c) categorical (boolean) user features
    tweet_texts     : dict {uid: [str]}    raw tweet texts per user (if tweets available)
    tweet_timestamps: dict {uid: [int]}    unix timestamps per user

    The tensor layout matches what UserFeatureEncoder and your dataloader expect.
    Numerical features are extracted from user.json and z-scored here so that
    your dataloader can feed them directly to NumericalEncoder.
    """
    print("\n[Stage 6]  Building graph_data.pt for ML pipeline")

    # ── ordered user index ────────────────────────────────────────────────
    user_ids = [record_id(u) for u in sub_users]
    uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    N = len(user_ids)
    print(f"  nodes: {N:,}")

    # ── labels ────────────────────────────────────────────────────────────
    label_map = {"bot": 1, "human": 0}
    label_lookup = dict(zip(
        label_df["id"].astype(str),
        label_df["label"].str.lower().map(label_map),
    ))
    labels = torch.tensor(
        [label_lookup.get(uid, -1) for uid in user_ids], dtype=torch.long
    )

    # ── split ─────────────────────────────────────────────────────────────
    split_map = {"train": 0, "valid": 1, "dev": 1, "val": 1, "test": 2}
    split_lookup = dict(zip(
        split_df["id"].astype(str),
        split_df["split"].str.lower().map(split_map),
    ))
    split_tensor = torch.tensor(
        [split_lookup.get(uid, 0) for uid in user_ids], dtype=torch.long
    )

    # ── user-user edge index (following / followers edges only) ───────────
    uu_edges = all_edges[all_edges["relation"].isin(USER_USER)].copy()
    src_idx = uu_edges["source"].map(uid_to_idx).dropna().astype(int)
    tgt_idx = uu_edges["target"].map(uid_to_idx).dropna().astype(int)
    valid_mask = (~uu_edges["source"].map(uid_to_idx).isna() &
                  ~uu_edges["target"].map(uid_to_idx).isna())
    src_idx = uu_edges.loc[valid_mask, "source"].map(uid_to_idx).astype(int).values
    tgt_idx = uu_edges.loc[valid_mask, "target"].map(uid_to_idx).astype(int).values
    edge_index = torch.tensor(
        np.stack([src_idx, tgt_idx], axis=0), dtype=torch.long
    )
    print(f"  user-user edges: {edge_index.shape[1]:,}")

    # ── numerical user features ───────────────────────────────────────────
    # Fields matching what NumericalEncoder expects (7 features by default)
    NUM_FIELDS = [
        "followers_count",      # 0
        "friends_count",        # 1  (following count)
        "listed_count",         # 2
        "favourites_count",     # 3
        "statuses_count",       # 4
        "default_profile",      # 5  (will be cast to float)
        "verified",             # 6
    ]
    CAT_FIELDS = [
        "default_profile",             # 0
        "default_profile_image",       # 1
        "verified",                    # 2
        "protected",                   # 3
        "geo_enabled",                 # 4
        "contributors_enabled",        # 5
        "is_translator",               # 6
        "is_translation_enabled",      # 7
        "following",                   # 8
        "follow_request_sent",         # 9
        "notifications",               # 10
    ]

    def _to_float(val, default=0.0):
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _to_bool(val):
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, str):
            return 1.0 if val.lower() in ("true", "1", "yes") else 0.0
        return float(bool(val)) if val is not None else 0.0

    user_lookup = {record_id(u): u for u in sub_users}

    num_feat = np.zeros((N, len(NUM_FIELDS)), dtype=np.float32)
    cat_feat = np.zeros((N, len(CAT_FIELDS)), dtype=np.float32)

    for i, uid in enumerate(user_ids):
        u = user_lookup.get(uid, {})
        for j, field in enumerate(NUM_FIELDS):
            num_feat[i, j] = _to_float(u.get(field, 0))
        for j, field in enumerate(CAT_FIELDS):
            cat_feat[i, j] = _to_bool(u.get(field, False))

    # z-score normalise numerical features (per-feature, clip to [-3, 3])
    mu  = num_feat.mean(axis=0, keepdims=True)
    std = num_feat.std(axis=0, keepdims=True) + 1e-8
    num_feat = np.clip((num_feat - mu) / std, -3.0, 3.0)

    user_num_feat = torch.tensor(num_feat, dtype=torch.float32)
    user_cat_feat = torch.tensor(cat_feat, dtype=torch.float32)
    print(f"  numerical features: {user_num_feat.shape}  "
          f"categorical features: {user_cat_feat.shape}")

    # ── tweet texts and timestamps (if tweets were collected) ─────────────
    tweet_texts:      Dict[str, list] = defaultdict(list)
    tweet_timestamps: Dict[str, list] = defaultdict(list)

    for tw in all_tweets:
        author = normalise_user_id(tw.get("author_id", tw.get("user", {}).get("id", "")))
        if author not in uid_to_idx:
            continue
        text = tw.get("full_text", tw.get("text", ""))
        tweet_texts[author].append(text)
        # created_at is a Twitter date string or unix int
        ts = tw.get("created_at", 0)
        if isinstance(ts, str):
            try:
                import datetime
                dt = datetime.datetime.strptime(ts, "%a %b %d %H:%M:%S +0000 %Y")
                ts = int(dt.timestamp())
            except Exception:
                ts = 0
        tweet_timestamps[author].append(int(ts))

    # ── save ──────────────────────────────────────────────────────────────
    out_path = output_dir / "graph_data.pt"
    torch.save({
        "user_ids":         user_ids,
        "labels":           labels,
        "split":            split_tensor,
        "edge_index":       edge_index,
        "user_num_feat":    user_num_feat,
        "user_cat_feat":    user_cat_feat,
        "tweet_texts":      dict(tweet_texts),
        "tweet_timestamps": dict(tweet_timestamps),
        # normalisation stats for downstream denorm if needed
        "num_feat_fields":  NUM_FIELDS,
        "cat_feat_fields":  CAT_FIELDS,
        "num_feat_mean":    mu.squeeze().tolist(),
        "num_feat_std":     (std - 1e-8).squeeze().tolist(),
    }, out_path)
    print(f"  wrote graph_data.pt  ({out_path.stat().st_size/1e6:.1f} MB)")

    # Quick sanity check
    n_train = int((split_tensor == 0).sum())
    n_val   = int((split_tensor == 1).sum())
    n_test  = int((split_tensor == 2).sum())
    n_bot   = int((labels == 1).sum())
    n_human = int((labels == 0).sum())
    print(f"\n  split:  train={n_train:,}  val={n_val:,}  test={n_test:,}")
    print(f"  labels: bot={n_bot:,}  human={n_human:,}  unlabelled={N-n_bot-n_human:,}")
    print(f"  bot ratio in labelled: {n_bot/(n_bot+n_human+1e-9):.1%}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_subset(args):
    t0 = time.time()
    input_dir  = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TwiBot-22 subset builder  –  mode={args.mode}")
    print(f"  input : {input_dir}")
    print(f"  output: {output_dir}")
    print(f"  users : {args.n_users:,}   seed: {args.seed}")
    print("=" * 60)

    # Stage 1: sample users
    sampled_users, label_df, split_df = stage1_sample_users(
        input_dir, args.n_users, args.seed
    )

    # Stage 2: vectorised edge scan
    all_edges, tweet_ids, list_ids, hashtag_ids = stage2_scan_edges(
        input_dir, sampled_users
    )

    # Stage 3: tweet streaming (graph mode skips this entirely)
    all_tweets: list = []
    if args.mode == "full":
        if not tweet_ids:
            print("\n[Stage 3]  No tweet IDs collected — skipping tweet scan.")
        else:
            all_tweets = stage3_stream_tweets(
                input_dir,
                tweet_ids,
                sampled_users,
                args.max_tweets_per_user,
                args.max_tweet_files,
            )
    else:
        print(f"\n[Stage 3]  Skipped (mode={args.mode}).  "
              f"Use --mode full to enable tweet streaming.")

    # Stage 4: filter entity JSON files
    sub_users, sub_lists, sub_hashtags = stage4_filter_entity_json(
        input_dir, sampled_users, list_ids, hashtag_ids
    )

    # Stage 5: write raw TwiBot-22 schema files
    stage5_write_raw_schema(
        output_dir, sub_users, all_tweets, sub_lists, sub_hashtags,
        all_edges, label_df, split_df, sampled_users, args.max_tweets_per_user,
    )

    # Stage 6: write ML-ready tensors
    stage6_write_graph_tensors(
        output_dir, sub_users, all_edges, label_df, split_df,
        sampled_users, all_tweets,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes.")
    print(f"  users         : {len(sub_users):,}")
    print(f"  tweets        : {len(all_tweets):,}")
    print(f"  edges (kept)  : {len(all_edges):,}")
    print(f"  output dir    : {output_dir}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Optimised TwiBot-22 subsetter for low-resource hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir",  type=Path, required=True,
                   help="Directory containing the full TwiBot-22 files.")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Directory for subset output.")
    p.add_argument("--n_users",    type=int,  default=10_000,
                   help="Number of users to sample.")
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--mode",       choices=["graph", "full"], default="graph",
                   help=(
                       "graph: fast, skips tweet files (recommended for initial runs). "
                       "full: also streams tweet_*.json files."
                   ))
    p.add_argument("--max_tweets_per_user", type=int, default=15,
                   help="[full mode] max tweets per user.")
    p.add_argument("--max_tweet_files",     type=int, default=1,
                   help="[full mode] max tweet_*.json files to scan. "
                        "1 is fast (covers most users). 0 = all files.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input_dir.exists():
        sys.exit(f"[ERROR] Input directory not found: {args.input_dir}")
    build_subset(args)
