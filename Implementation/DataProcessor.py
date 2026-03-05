import json
import numpy as np
import torch
from datetime import datetime
from torch_geometric.data import Data
import os

def extract_static_features(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Reference date for TwiBot-20 (October 1, 2020)
    ref_date = datetime(2020, 10, 1)
    
    all_static_vectors = []
    user_ids = []
    labels = []

    for user in data:
        profile = user['profile']
        
        # (Log Transformed)
        followers = np.log1p(float(profile.get('followers_count', 0)))
        friends = np.log1p(float(profile.get('friends_count', 0)))
        statuses = np.log1p(float(profile.get('statuses_count', 0)))
        favs = np.log1p(float(profile.get('favourites_count', 0)))
        listed = np.log1p(float(profile.get('listed_count', 0)))

        # Binary Features 
        def to_binary(val):
            if isinstance(val, str):
                return 1 if "true" in val.lower() else 0
            return 1 if val else 0

        is_verified = to_binary(profile.get('verified', False))
        is_default_profile = to_binary(profile.get('default_profile', False))
        is_geo_enabled = to_binary(profile.get('geo_enabled', False))

        #  Content Length Features
        # Bio length and screen name length can distinguish procedurally generated accounts
        bio = profile.get('description', "")
        bio_len = len(bio) if bio is not None else 0
        screen_name_len = len(profile.get('screen_name', ""))

        # Account Age Calculation 
        # Twitter Date Format: "Tue Nov 18 10:27:25 +0000 2008"
        created_at_str = profile.get('created_at', "")
        try:
            
            clean_date_str = created_at_str.strip()
            dt = datetime.strptime(clean_date_str, '%a %b %d %H:%M:%S +0000 %Y')
            age_days = (ref_date - dt).days
            account_age = np.log1p(max(0, age_days)) 
        except:
            # Fallback if date is missing or malformed
            account_age = 0

        # Build Final Vector
        feature_vector = [
            followers, friends, statuses, favs, listed, 
            is_verified, is_default_profile, is_geo_enabled,
            bio_len, screen_name_len, account_age
        ]
        
        all_static_vectors.append(feature_vector)
        user_ids.append(user['ID'])
        labels.append(int(user.get('label', -1)))

    feature_matrix = torch.tensor(all_static_vectors, dtype=torch.float32)
    return feature_matrix, user_ids, labels

def extract_behavioral_features(user_obj, account_age_days):
    tweets = user_obj.get('tweet')
    if tweets is None:
        tweets = []
    num_tweets = len(tweets)
    
    # Return 4 zeros to match the expected dimension if no tweets exist
    if num_tweets == 0:
        return [0.0, 0.0, 0.0, 0.0]

    # 1. RT Rate: How many are Retweets?
    rt_count = sum(1 for t in tweets if t and t.strip().startswith("RT @"))
    rt_rate = rt_count / num_tweets
    
    # 2. Link Rate: How many contain 'http'?
    link_count = sum(1 for t in tweets if t and "http" in t.lower())
    link_rate = link_count / num_tweets
    
    # 3. Mention Rate: Average @mentions per tweet
    mention_count = sum(t.count("@") for t in tweets if t)
    avg_mentions = mention_count / num_tweets
    
    # 4. Status Velocity: Total tweets / Account Age
    profile_total_tweets = float(user_obj['profile'].get('statuses_count', 0))
    velocity = profile_total_tweets / (max(1, account_age_days)) # Avoid div by zero
    log_velocity = np.log1p(velocity)

    return [rt_rate, link_rate, avg_mentions, log_velocity]


def safe_load_json(file_path):
    if file_path is None:
        return []
    
    file_name = os.path.basename(file_path)
    print(f"Attempting to load {file_name}...")

    # STRATEGY 1: The Fast Way (Standard Load)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Successfully loaded {len(data)} users via Fast-Load.")
            return data
    except Exception as e:
        print(f"Fast-Load failed ({e}). Switching to Robust-Streaming...")

    # STRATEGY 2: The Robust Way (Line-by-Line with Progress)
    data = []
    buffer = ""
    bracket_level = 0
    line_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % 50000 == 0:
                    print(f"  Reading line {line_count}... Found {len(data)} users so far.")
                
                line_str = line.strip()
                if not line_str or line_str in ['[', ']', '],']:
                    continue
                
                buffer += line
                bracket_level += line.count('{')
                bracket_level -= line.count('}')
                
                if bracket_level == 0 and buffer.strip():
                    try:
                        clean_obj = buffer.strip().rstrip(',')
                        data.append(json.loads(clean_obj))
                    except json.JSONDecodeError:
                        pass
                    buffer = "" 
        
        print(f"Successfully loaded {len(data)} users via Streaming.")
        return data
    except Exception as e:
        print(f"Critical error: {e}")
        return []
    
def extract_tweet_sequences(user_obj, max_tweets=20):
    """
    Creates a numerical sequence representing the 'pulse' of a user's 
    recent tweets. Each tweet is a 4-dim vector.
    """
    raw_tweets = user_obj.get('tweet', [])
    if raw_tweets is None: raw_tweets = []
    
    # Filter out None/Empty and take the most recent ones
    tweets = [t for t in raw_tweets if t][:max_tweets]
    sequence = []
    
    for t in tweets:
        # Normalized Length (Twitter limit is 280)
        length = min(1.0, len(t) / 280.0)
        # Contains Link
        has_link = 1.0 if "http" in t.lower() else 0.0
        # Is a Retweet
        is_rt = 1.0 if t.strip().startswith("RT @") else 0.0
        # Mention Density
        mentions = min(1.0, t.count("@") / 5.0)
        
        sequence.append([length, has_link, is_rt, mentions])
    
    # PADDING: If user has fewer than max_tweets, fill with zeros
    while len(sequence) < max_tweets:
        sequence.append([0.0, 0.0, 0.0, 0.0])
        
    return sequence

def build_complete_dataset(train_path, dev_path=None, test_path=None):
    """
    Main pipeline: Loads JSON data, builds a global ID map, extracts features, 
    constructs the graph adjacency matrix, and applies train/val/test splits.
    
    Returns:
        Data: A PyTorch Geometric Data object containing the graph.
        dict: A mapping of Twitter IDs to index integers.
    """
    train_raw = safe_load_json(train_path)
    dev_raw = safe_load_json(dev_path) if dev_path else []
    test_raw = safe_load_json(test_path) if test_path else []
    
    all_users = train_raw + dev_raw + test_raw
    ref_date = datetime(2020, 10, 1)
    
    id_map = {}
    feature_list = []
    sequence_list = []
    labels = []

    for i, user in enumerate(all_users):
        u_id = user['ID']
        id_map[u_id] = i
        profile = user['profile']
        
        #  Extraction Logic (Numerical) 
        followers = np.log1p(float(profile.get('followers_count', 0)))
        friends = np.log1p(float(profile.get('friends_count', 0)))
        statuses = np.log1p(float(profile.get('statuses_count', 0)))
        favs = np.log1p(float(profile.get('favourites_count', 0)))
        listed = np.log1p(float(profile.get('listed_count', 0)))
        
        # Extraction Logic (Binary)
        # Robust check for "True " vs "True" vs True
        is_verified = 1 if "true" in str(profile.get('verified', "")).lower() else 0
        is_default = 1 if "true" in str(profile.get('default_profile', "")).lower() else 0
        is_geo = 1 if "true" in str(profile.get('geo_enabled', "")).lower() else 0
        
        # Extraction Logic (Text-based/Behavioral) 
        bio_len = len(profile.get('description', "") or "")
        sn_len = len(profile.get('screen_name', "") or "")
        
        # Age Calculation
        raw_age_days = max(1, (ref_date - dt).days) # Default fallback
        try:
            dt = datetime.strptime(profile['created_at'].strip(), '%a %b %d %H:%M:%S +0000 %Y')
            account_age = np.log1p(max(0, (ref_date - dt).days))
        except:
            account_age = 0

        # Behavioral Features (The "Pulse")
        behavioral = extract_behavioral_features(user, raw_age_days)
    
        # Combine into ONE vector and append ONCE
        full_feature_vector = [
            followers, friends, statuses, favs, listed, 
            is_verified, is_default, is_geo, bio_len, sn_len, account_age
        ] + behavioral
    
        feature_list.append(full_feature_vector)
        user_sequence = extract_tweet_sequences(user, max_tweets=20) #mulbot
        sequence_list.append(user_sequence)
        labels.append(int(user.get('label', -1)))

    # 3. Build Edges using the id_map created above
    edge_list = []
    for user in all_users:
        u_idx = id_map[user['ID']]
        neighbors = user.get('neighbor')
        if neighbors:
            for target_id in (neighbors.get('following') or []):
                if target_id in id_map:
                    edge_list.append([u_idx, id_map[target_id]])
            for target_id in (neighbors.get('follower') or []):
                if target_id in id_map:
                    edge_list.append([id_map[target_id], u_idx])

    x_static = torch.tensor(feature_list, dtype=torch.float32)
    x_sequence = torch.tensor(sequence_list, dtype=torch.float32) # [N, 20, 4]
    y = torch.tensor(labels, dtype=torch.long)
    # If no edges found, edge_index needs a safe shape
    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    

    # 7. Masking
    num_nodes = len(all_users)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    n_train, n_dev = len(train_raw), len(dev_raw)
    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_dev] = True
    test_mask[n_train + n_dev :] = True

    graph_data = Data(x=x_static, edge_index=edge_index, y=y, 
                      train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    graph_data.x_sequence = x_sequence # Add the MULBOT input here
    
    return graph_data, id_map



base_path = r"D:\3rd year project\Implementation\archive"

train_file = os.path.join(base_path, "train.json")
dev_file = os.path.join(base_path, "dev.json")
test_file = os.path.join(base_path, "test.json")


print(f"Checking for files in: {base_path}...")
files_exist = True
for f in [train_file, dev_file, test_file]:
    if not os.path.exists(f):
        print(f"!! Error: Missing file: {f}")
        files_exist = False

if files_exist:
    print("Loading and processing data (TRAIN ONLY mode)...")

    data, id_map = build_complete_dataset(train_file, None, None)
    print("\n--- Data Sanity Check ---")
    print(f"Total Nodes (Users): {data.num_nodes}")
    print(f"Feature Matrix Shape: {data.x.shape}")
    print(f"Edge Index Shape: {data.edge_index.shape}")
    print(f"Temporal Sequence Shape: {data.x_sequence.shape}")
    print(f"Number of Labels: {data.y.shape[0]}")

    if data.edge_index.shape[1] > 0:
        print(f"Success! Found {data.edge_index.shape[1]} neighbor connections.")
    else:
        print("Warning: No edges found. Check if IDs match in the neighbor fields.")

    if torch.isnan(data.x).any():
        print("CRITICAL: NaNs found in feature matrix!")
    else:
        print("Feature matrix is clean and ready for training.")
else:
    print("Execution halted: Please fix the file paths above.")

processed_path = os.path.join(base_path, "processed_data.pt")
id_map_path = os.path.join(base_path, "id_map.json")

print(f"\nSaving processed graph data to {processed_path}...")
torch.save(data, processed_path)

print(f"Saving ID map to {id_map_path}...")
with open(id_map_path, 'w') as f:
    json.dump(id_map, f)

print("Phase 1 Complete: Ingredients are ready and stored.")
