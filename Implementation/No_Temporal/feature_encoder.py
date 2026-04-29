"""
BotDCGC – Section 4.1: User Attribute Feature Encoding
Encodes numerical, categorical, description, and tweet features,
then fuses them with Bi-LSTM into a single node embedding X.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  Numerical feature encoder
# ---------------------------------------------------------------------------
class NumericalEncoder(nn.Module):
    """
    z-score-normalised numerical features → FC → R^{D/4}.
    Normalisation is handled outside (in the dataset); the FC projects
    the raw (already-normalised) floats to the embedding sub-space.
    """
    def __init__(self, num_features: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, num_features)
        return self.fc(x)                                         # (B, D/4)


# ---------------------------------------------------------------------------
# Categorical feature encoder
# ---------------------------------------------------------------------------
class CategoricalEncoder(nn.Module):
    """
    One-hot (Boolean) categorical features → FC → R^{D/4}.
    """
    def __init__(self, cat_features: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cat_features, out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, cat_features)
        return self.fc(x)                                         # (B, D/4)


# ---------------------------------------------------------------------------
# Description feature encoder  (RoBERTa pre-encoding assumed)
# ---------------------------------------------------------------------------
class DescriptionEncoder(nn.Module):
    """
    Accepts the RoBERTa CLS embedding of a user's description
    (shape: B × Dd) and projects it to R^{D/4} via a learnable FC.

    In production: call a frozen RoBERTa model upstream and pass its
    output here.  During testing a random Dd-dim vector is accepted.
    """
    def __init__(self, roberta_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(roberta_dim, out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, d: torch.Tensor) -> torch.Tensor:          # (B, Dd)
        return self.fc(d)                                         # (B, D/4)


# ---------------------------------------------------------------------------
# Tweet feature encoder
# ---------------------------------------------------------------------------
class TweetEncoder(nn.Module):
    """
    For each tweet the words are pre-encoded with RoBERTa → t^j_i.
    An LSTM reads the word sequence; its hidden states are averaged
    per tweet, then all tweet averages are averaged to give r_t ∈ R^{D/4}.

    Input:  tweet_emb  (B, M, max_words, roberta_dim)
            tweet_len  (B, M) – actual word count per tweet (for packing)
    """
    def __init__(self, roberta_dim: int, lstm_hidden: int, out_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(roberta_dim, lstm_hidden,
                            batch_first=True, bidirectional=False)
        self.fc   = nn.Linear(lstm_hidden, out_dim)
        self.act  = nn.LeakyReLU(0.2)

    def forward(self,
                tweet_emb: torch.Tensor,   # (B, M, Oj, roberta_dim)
                tweet_len: torch.Tensor    # (B, M)  – words per tweet
                ) -> torch.Tensor:
        B, M, Oj, Rd = tweet_emb.shape

        # Flatten (B, M) to process all tweets in one LSTM pass
        x = tweet_emb.view(B * M, Oj, Rd)                        # (B*M, Oj, Rd)
        lengths = tweet_len.view(B * M).clamp(min=1).cpu()

        packed   = nn.utils.rnn.pack_padded_sequence(
                        x, lengths, batch_first=True, enforce_sorted=False)
        out, _   = self.lstm(packed)
        out, _   = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # out: (B*M, Oj, hidden)

        # Average over words (masking pad positions)
        T = out.size(1)
        mask = (torch.arange(T, device=out.device)[None, :] <
            lengths[:, None].to(out.device)).float()
        tweet_rep = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        # tweet_rep: (B*M, hidden)

        tweet_rep = tweet_rep.view(B, M, -1)                      # (B, M, hidden)
        tweet_mask = (tweet_len > 0).float().to(tweet_rep.device)                   # (B, M)
        user_rep = (tweet_rep * tweet_mask.unsqueeze(-1)).sum(dim=1) / \
           tweet_mask.sum(dim=1, keepdim=True).clamp(min=1)                         # (B, hidden)
        return self.act(self.fc(user_rep))                        # (B, D/4)


# ---------------------------------------------------------------------------
# Feature fusion with Bi-LSTM
# ---------------------------------------------------------------------------
class FeatureFusion(nn.Module):
    """
    Concatenates the four sub-embeddings (each D/4) into a sequence of
    length 4, then runs a Bi-LSTM over them and mean-pools to produce the
    final node embedding x_i ∈ R^D.

    The paper encodes heterogeneous feature types as a *sequence* with
    Bi-LSTM so that cross-feature interactions are captured (Eq. 7).
    """
    def __init__(self, sub_dim: int, out_dim: int):
        super().__init__()
        # sub_dim = D/4; the Bi-LSTM reads a 4-step sequence of D/4-dim vectors
        assert out_dim % 2 == 0, "out_dim must be even for Bi-LSTM"
        self.bilstm = nn.LSTM(sub_dim, out_dim // 2,
                              batch_first=True, bidirectional=True)

    def forward(self,
                r_n: torch.Tensor,   # (B, D/4)
                r_c: torch.Tensor,   # (B, D/4)
                r_d: torch.Tensor,   # (B, D/4)
                r_t: torch.Tensor,   # (B, D/4)
                ) -> torch.Tensor:
        # Stack as a 4-step sequence: (B, 4, D/4)
        seq = torch.stack([r_n, r_c, r_d, r_t], dim=1)
        out, _ = self.bilstm(seq)                                 # (B, 4, D)
        x = out.mean(dim=1)                                       # (B, D)
        return x


# ---------------------------------------------------------------------------
# Top-level user feature encoder
# ---------------------------------------------------------------------------
class UserFeatureEncoder(nn.Module):
    """
    Shared user feature encoder used by both BotDCGC and CACL.

    Encodes:
    - numerical features
    - categorical features
    - description embeddings
    - tweet embeddings

    and fuses them into a shared user representation X_shared of shape (N, D).
    """
    def __init__(self,
                 num_features: int = 7,
                 cat_features: int = 11,
                 roberta_dim: int = 768,
                 lstm_hidden: int = 128,
                 embed_dim: int = 256):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        sub = embed_dim // 4
        self.num_enc = NumericalEncoder(num_features, sub)
        self.cat_enc = CategoricalEncoder(cat_features, sub)
        self.des_enc = DescriptionEncoder(roberta_dim, sub)
        self.twt_enc = TweetEncoder(roberta_dim, lstm_hidden, sub)
        self.fusion = FeatureFusion(sub, embed_dim)

    def forward(self,
                num: torch.Tensor,
                cat: torch.Tensor,
                desc_emb: torch.Tensor,
                tweet_emb: torch.Tensor,
                tweet_len: torch.Tensor,
                return_parts: bool = False):
        r_n = self.num_enc(num)
        r_c = self.cat_enc(cat)
        r_d = self.des_enc(desc_emb)
        r_t = self.twt_enc(tweet_emb, tweet_len)

        x_shared = self.fusion(r_n, r_c, r_d, r_t)

        if return_parts:
            return {
                "x_shared": x_shared,
                "r_num": r_n,
                "r_cat": r_c,
                "r_desc": r_d,
                "r_tweet": r_t,
            }

        return x_shared