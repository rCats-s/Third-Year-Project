import torch

data = torch.load(r"D:\3rd year project\Implementation\Datasets\twibot22_smoke\graph_data.pt")

nonzero_users = 0
sample_good = None

for uid, ts_list in data["tweet_timestamps"].items():
    valid = [t for t in ts_list if isinstance(t, (int, float)) and t > 0]
    if valid:
        nonzero_users += 1
        if sample_good is None:
            sample_good = (uid, valid[:10])

print("users with at least one nonzero timestamp:", nonzero_users)

if sample_good:
    print("example valid user:", sample_good[0])
    print("example timestamps:", sample_good[1])
else:
    print("No valid timestamps found.")