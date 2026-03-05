import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, Reshape

def build_mts(df, feature_cols, time_col="date", user_col="user_id"):
    df[time_col] = pd.to_datetime(df[time_col])
    df["day"] = df[time_col].dt.date

    # Aggregate per user per day
    daily = (
        df.groupby([user_col, "day"])[feature_cols]
        .sum()
        .reset_index()
    )

    # Pivot to MTS format
    users = daily[user_col].unique()
    all_days = sorted(daily["day"].unique())

    user_index = {u: i for i, u in enumerate(users)}
    day_index = {d: i for i, d in enumerate(all_days)}

    N = len(users)
    T = len(all_days)
    D = len(feature_cols)

    mts = np.full((N, T, D), -1.0)

    for _, row in daily.iterrows():
        i = user_index[row[user_col]]
        t = day_index[row["day"]]
        mts[i, t, :] = row[feature_cols].values

    return mts, users

def normalize_mts(mts):
    N, T, D = mts.shape
    mts_reshaped = mts.reshape(-1, D)

    scaler = MinMaxScaler()
    mts_scaled = scaler.fit_transform(mts_reshaped)

    return mts_scaled.reshape(N, T, D)

def build_lstm_autoencoder(T, D, latent_dim=300):

    inputs = Input(shape=(T, D))

    # Encoder
    encoded = LSTM(1, return_sequences=True)(inputs)
    flat = Flatten()(encoded)
    latent = Dense(latent_dim)(flat)

    # Decoder
    decoded_dense = Dense(T)(latent)
    reshaped = Reshape((T, 1))(decoded_dense)
    decoded = LSTM(D, return_sequences=True)(reshaped)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, latent)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0002),
        loss="mse"
    )

    return autoencoder, encoder

def train_autoencoder(mts, epochs=50):

    N, T, D = mts.shape
    X_train, X_test = train_test_split(mts, test_size=0.3, random_state=42)

    autoencoder, encoder = build_lstm_autoencoder(T, D)

    autoencoder.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=epochs,
        batch_size=len(X_train),
        verbose=1
    )

    latent_vectors = encoder.predict(mts)
    return latent_vectors

def hierarchical_clustering(latent_vectors, n_clusters=2): # This is for binary
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(latent_vectors)
    return labels

def dbscan_clustering(latent_vectors, eps=0.5, min_samples=5): # This is for multiclass
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(latent_vectors)
    return labels

def evaluate(true_labels, cluster_labels):

    print("Accuracy:", accuracy_score(true_labels, cluster_labels))
    print("F1 Score:", f1_score(true_labels, cluster_labels))
    print("\nClassification Report:\n")
    print(classification_report(true_labels, cluster_labels))

# Feature list from paper
features = [
    "retweet_count",
    "reply_count",
    "favorite_count",
    "num_mentions"
]

# Build MTS
mts, users = build_mts( features)

# Normalize
mts = normalize_mts(mts)

# Train autoencoder
latent_vectors = train_autoencoder(mts, epochs=100)

# Cluster
cluster_labels = hierarchical_clustering(latent_vectors)

# Evaluate
evaluate(cluster_labels)

