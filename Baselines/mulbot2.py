import numpy as np
import pandas as pd

def extract_mts(
    df,
    user_col="user_id",
    time_col="created_at",
    feature_cols=None,
    start_date=None,
    end_date=None
):
    """
    Extract Multivariate Time Series (MTS) with daily granularity.

    Returns:
        mts: numpy array of shape (N, T, D)
        users: list of user ids (length N)
        dates: list of timestamps (length T)
    """

    if feature_cols is None:
        feature_cols = [
            "num_urls",
            "num_hashtags",
            "num_mentions",
            "retweet_count",
            "reply_count",
            "favorite_count",
        ]

    # Ensure datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Set date boundaries
    if start_date is None:
        start_date = df[time_col].min().floor("D")
    if end_date is None:
        end_date = df[time_col].max().floor("D")

    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    users = df[user_col].unique()

    N = len(users)
    T = len(all_dates)
    D = len(feature_cols)

    # Initialize with -1 (special value for no tweets)
    mts = np.full((N, T, D), -1, dtype=float)

    # Group by user and day
    df["date"] = df[time_col].dt.floor("D")
    grouped = df.groupby([user_col, "date"])

    for n, user in enumerate(users):
        user_data = df[df[user_col] == user]

        if user_data.empty:
            continue

        daily_groups = user_data.groupby("date")

        for t, date in enumerate(all_dates):
            if date in daily_groups.groups:
                day_df = daily_groups.get_group(date)

                # TW^n_t != 0 case
                for d, feature in enumerate(feature_cols):
                    mts[n, t, d] = day_df[feature].sum()
            # else: remains -1 automatically

    return mts, users, all_dates