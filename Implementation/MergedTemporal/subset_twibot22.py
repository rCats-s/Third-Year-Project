"""
subset_twibot22.py
==================
Create a laptop-friendly subset of TwiBot-22 that preserves the original
file schema so it is a drop-in replacement for your dataloader.

Improved strategy
-----------------
1. Stratified sample users by BOTH split and label.
2. Single pass through edge.csv to collect only entities connected to sampled users.
3. Keep user-user edges ONLY when both ends are sampled users.
4. Keep user→tweet/list/hashtag edges ONLY when the sampled user is the source.
5. Stream each tweet_i.json with ijson.
6. Filter user.json, list.json, hashtag.json.
7. Re-write all output files with the same schema.

Requirements
------------
    pip install ijson pandas tqdm

Usage
-----
    python subset_twibot22.py \
        --input_dir  /path/to/twibot22 \
        --output_dir /path/to/subset   \
        --n_users    15000             \
        --seed       42                \
        --max_tweets_per_user 20
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import ijson
except ImportError:
    sys.exit(
        "[ERROR] ijson not installed. Run: pip install ijson\n"
        "ijson is required for memory-efficient streaming of large JSON files."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_json(path: Path, desc: str) -> list:
    """Load a JSON array from disk (for files that fit in RAM)."""
    print(f"  Loading {desc} from {path.name} …", end=" ", flush=True)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"{len(data):,} records")
    return data


def stream_tweet_file(path: Path, wanted_ids: set, max_per_user: dict) -> list:
    """
    Stream one tweet_i.json file with ijson.
    Only yields tweet objects whose 'id' is in wanted_ids and whose
    author's per-user budget has not been exhausted.
    """
    kept = []
    file_size = os.path.getsize(path)

    with open(path, "rb") as f:
        items = ijson.items(f, "item")
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"    streaming {path.name}",
            leave=False,
        ) as bar:
            for tweet in items:
                tid = str(tweet.get("id", tweet.get("id_str", "")))
                if tid in wanted_ids:
                    author = str(
                        tweet.get("author_id", tweet.get("user", {}).get("id", ""))
                    )
                    if max_per_user.get(author, 0) > 0:
                        kept.append(tweet)
                        max_per_user[author] -= 1
                bar.update(0)

    return kept


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    if "uid" in df.columns:
        rename_map["uid"] = "id"
    if "source_id" in df.columns:
        rename_map["source_id"] = "source"
    if "target_id" in df.columns:
        rename_map["target_id"] = "target"
    if "src" in df.columns:
        rename_map["src"] = "source"
    if "tgt" in df.columns:
        rename_map["tgt"] = "target"
    if "relation_type" in df.columns:
        rename_map["relation_type"] = "relation"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_subset(
    input_dir: Path,
    output_dir: Path,
    n_users: int,
    seed: int,
    max_tweets_per_user: int,
    max_tweet_files: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # ── Stage 1: stratified user sampling by split + label ───────────────
    print("\n[Stage 1] Stratified user sampling by split + label")

    label_df = pd.read_csv(input_dir / "label.csv")
    split_df = pd.read_csv(input_dir / "split.csv")

    label_df = normalise_columns(label_df)
    split_df = normalise_columns(split_df)

    label_df["id"] = label_df["id"].astype(str)
    split_df["id"] = split_df["id"].astype(str)
    label_df["label"] = label_df["label"].astype(str).str.lower()
    split_df["split"] = split_df["split"].astype(str).str.lower()

    # normalise valid/dev naming
    split_df["split"] = split_df["split"].replace({
        "dev": "valid",
        "val": "valid",
    })

    merged = label_df.merge(split_df, on="id", how="inner")

    # group sizes
    group_sizes = (
        merged.groupby(["split", "label"])
        .size()
        .reset_index(name="count")
    )

    total_available = int(group_sizes["count"].sum())
    if n_users > total_available:
        print(f"  Requested {n_users:,} users, but only {total_available:,} available.")
        n_users = total_available

    # allocate sample targets proportionally across split+label groups
    group_sizes["target_float"] = group_sizes["count"] / total_available * n_users
    group_sizes["target"] = group_sizes["target_float"].astype(int)

    # distribute remainder by largest fractional parts
    remainder = n_users - int(group_sizes["target"].sum())
    if remainder > 0:
        group_sizes["frac"] = group_sizes["target_float"] - group_sizes["target"]
        group_sizes = group_sizes.sort_values("frac", ascending=False).reset_index(drop=True)
        for i in range(remainder):
            group_sizes.loc[i, "target"] += 1

    selected_parts = []
    for _, row in group_sizes.iterrows():
        split_name = row["split"]
        label_name = row["label"]
        target = int(row["target"])

        group = merged[(merged["split"] == split_name) & (merged["label"] == label_name)]
        ids = group["id"].tolist()

        if target > len(ids):
            target = len(ids)

        sampled = rng.sample(ids, target) if target > 0 else []
        selected_parts.append(group[group["id"].isin(sampled)])

        print(
            f"  {split_name:>5} | {label_name:>5} | "
            f"available={len(ids):>6,} | sampled={target:>6,}"
        )

    selected_df = pd.concat(selected_parts, ignore_index=True)
    sampled_users = set(selected_df["id"].astype(str))

    print(f"\n  Total sampled users: {len(sampled_users):,}")

    # ── Stage 2: collect entity IDs via relation-aware edge filtering ────
    print("\n[Stage 2] Scanning edge.csv with relation-aware filtering")

    USER_USER = {"following", "followers", "followed"}
    USER_TO_TWEET = {"pinned", "post", "retweeted", "like", "quoted", "replied", "mentioned"}
    USER_TO_LIST = {"own", "membership"}
    USER_TO_HASHTAG = {"discuss", "contain"}

    tweet_ids = set()
    list_ids = set()
    hashtag_ids = set()
    kept_edges = []

    edge_candidates = sorted(input_dir.glob("edge*.csv"))
    if not edge_candidates:
        sys.exit("[ERROR] No edge CSV file found (expected something like edge.csv or edge-001.csv).")
    edge_path = edge_candidates[0]

    with open(edge_path, encoding="utf-8") as f:
        reader = pd.read_csv(f, chunksize=500_000, dtype=str)

        for chunk in tqdm(reader, desc="  reading edge.csv chunks"):
            chunk = normalise_columns(chunk)
            chunk["source"] = chunk["source"].astype(str)
            chunk["target"] = chunk["target"].astype(str)
            chunk["relation"] = chunk["relation"].astype(str)

            for _, row in chunk.iterrows():
                src = row["source"]
                tgt = row["target"]
                rel = row["relation"]

                # user-user edges: keep only if BOTH ends are sampled users
                if rel in USER_USER:
                    if src in sampled_users and tgt in sampled_users:
                        kept_edges.append({
                            "source_id": src,
                            "target_id": tgt,
                            "relation": rel,
                        })

                # user -> tweet edges: keep only if sampled user is source
                elif rel in USER_TO_TWEET:
                    if src in sampled_users:
                        kept_edges.append({
                            "source_id": src,
                            "target_id": tgt,
                            "relation": rel,
                        })
                        tweet_ids.add(tgt)

                # user -> list edges
                elif rel in USER_TO_LIST:
                    if src in sampled_users:
                        kept_edges.append({
                            "source_id": src,
                            "target_id": tgt,
                            "relation": rel,
                        })
                        list_ids.add(tgt)

                # user -> hashtag edges
                elif rel in USER_TO_HASHTAG:
                    if src in sampled_users:
                        kept_edges.append({
                            "source_id": src,
                            "target_id": tgt,
                            "relation": rel,
                        })
                        hashtag_ids.add(tgt)

                # all other relations ignored for now

    print(
        f"  Collected {len(tweet_ids):,} tweet IDs | "
        f"{len(list_ids):,} list IDs | "
        f"{len(hashtag_ids):,} hashtag IDs | "
        f"{len(kept_edges):,} kept edges"
    )

    # ── Stage 3: stream tweet files ───────────────────────────────────────
    print("\n[Stage 3] Streaming tweet_*.json files")

    budget = defaultdict(lambda: max_tweets_per_user)
    tweet_files = sorted(input_dir.glob("tweet_*.json"))
    if max_tweet_files > 0:
        tweet_files = tweet_files[:max_tweet_files]

    if not tweet_files:
        print("  [WARN] No tweet_*.json files found – skipping tweet extraction.")
        all_tweets = []
    else:
        all_tweets = []
        for tfile in tweet_files:
            chunk = stream_tweet_file(tfile, tweet_ids, budget)
            all_tweets.extend(chunk)
            print(f"    {tfile.name}: kept {len(chunk):,} tweets (running total: {len(all_tweets):,})")

    print(f"  Total tweets kept: {len(all_tweets):,}")

    # ── Stage 4: filter user / list / hashtag JSON ────────────────────────
    print("\n[Stage 4] Filtering user / list / hashtag JSON")

    raw_users = load_json(input_dir / "user.json", "users")
    subset_users = [u for u in raw_users if str(u.get("id", u.get("id_str", ""))) in sampled_users]
    print(f"  Filtered to {len(subset_users):,} user records")

    list_file = input_dir / "list.json"
    if list_file.exists():
        raw_lists = load_json(list_file, "lists")
        subset_lists = [l for l in raw_lists if str(l.get("id", l.get("id_str", ""))) in list_ids]
        print(f"  Filtered to {len(subset_lists):,} list records")
    else:
        subset_lists = []
        print("  list.json not found – skipping")

    hashtag_file = input_dir / "hashtag.json"
    if hashtag_file.exists():
        raw_hashtags = load_json(hashtag_file, "hashtags")
        subset_hashtags = [h for h in raw_hashtags if str(h.get("id", h.get("id_str", h.get("tag", "")))) in hashtag_ids]
        print(f"  Filtered to {len(subset_hashtags):,} hashtag records")
    else:
        subset_hashtags = []
        print("  hashtag.json not found – skipping")

    # ── Stage 5: write output ─────────────────────────────────────────────
    print("\n[Stage 5] Writing subset files to", output_dir)

    def write_json(data: list, fname: str):
        out_path = output_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Wrote {fname} ({len(data):,} records, {size_mb:.1f} MB)")

    write_json(subset_users, "user.json")
    write_json(all_tweets, "tweet_0.json")
    write_json(subset_lists, "list.json")
    write_json(subset_hashtags, "hashtag.json")

    edge_out = output_dir / "edge.csv"
    pd.DataFrame(kept_edges).to_csv(edge_out, index=False)
    print(f"  Wrote edge.csv ({len(kept_edges):,} rows, {edge_out.stat().st_size / 1e6:.1f} MB)")

    subset_label = label_df[label_df["id"].isin(sampled_users)].copy()
    subset_label.to_csv(output_dir / "label.csv", index=False)
    print(f"  Wrote label.csv ({len(subset_label):,} rows)")

    subset_split = split_df[split_df["id"].isin(sampled_users)].copy()
    subset_split.to_csv(output_dir / "split.csv", index=False)
    print(f"  Wrote split.csv ({len(subset_split):,} rows)")

    print("\n" + "=" * 56)
    print("Subset complete.")
    print(f"  Users   : {len(subset_users):,}")
    print(f"  Tweets  : {len(all_tweets):,} (max {max_tweets_per_user} / user)")
    print(f"  Lists   : {len(subset_lists):,}")
    print(f"  Hashtags: {len(subset_hashtags):,}")
    print(f"  Edges   : {len(kept_edges):,}")
    print(f"  Output  : {output_dir}")
    print("=" * 56)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Create a stratified TwiBot-22 subset for laptop-scale experiments."
    )
    p.add_argument(
        "--input_dir", type=Path, required=True,
        help="Path to the directory containing the full TwiBot-22 dataset files."
    )
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory where the subset files will be written."
    )
    p.add_argument(
        "--n_users", type=int, default=500,
        help="Number of users to sample."
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    p.add_argument(
        "--max_tweets_per_user", type=int, default=5,
        help="Maximum tweets to keep per sampled user."
    )
    p.add_argument(
        "--max_tweet_files", type=int, default=1,
        help="Maximum number of tweet_*.json files to scan for quick verification. Use 0 or negative for all files."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input_dir.exists():
        sys.exit(f"[ERROR] Input directory not found: {args.input_dir}")

    build_subset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_users=args.n_users,
        seed=args.seed,
        max_tweets_per_user=args.max_tweets_per_user,
        max_tweet_files=args.max_tweet_files,
    )