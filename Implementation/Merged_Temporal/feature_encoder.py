"""
BotDCGC / CACL – Section 4.1: User Attribute Feature Encoding

Encodes:
- numerical features   (static + temporal numerical statistics)
- categorical features (static + temporal categorical / boolean flags)
- description features
- tweet text features

Then fuses them with a Bi-LSTM into a single shared node embedding X_shared.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Numerical feature encoder
# ---------------------------------------------------------------------------
class NumericalEncoder(nn.Module):
    
    def __init__(self, num_features: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)   # (B, D/4)


# ---------------------------------------------------------------------------
# Categorical feature encoder
# ---------------------------------------------------------------------------
class CategoricalEncoder(nn.Module):
    
    def __init__(self, cat_features: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cat_features, out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)   # (B, D/4)


# ---------------------------------------------------------------------------
# 4.1.3  Description feature encoder
# ---------------------------------------------------------------------------
class DescriptionEncoder(nn.Module):
    """
    Accepts the RoBERTa CLS embedding of a user's description
    (shape: B x Dd) and projects it to R^{D/4}.
    """
    def __init__(self, roberta_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(roberta_dim, out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return self.fc(d)   # (B, D/4)


# ---------------------------------------------------------------------------
# Tweet text encoder
# ---------------------------------------------------------------------------
class TweetEncoder(nn.Module):
    
    def __init__(self, roberta_dim: int, lstm_hidden: int, out_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            roberta_dim,
            lstm_hidden,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(lstm_hidden, out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        tweet_emb: torch.Tensor,   # (B, M, Oj, roberta_dim)
        tweet_len: torch.Tensor    # (B, M) token counts
    ) -> torch.Tensor:
        B, M, Oj, Rd = tweet_emb.shape

        x = tweet_emb.view(B * M, Oj, Rd)
        lengths = tweet_len.view(B * M).clamp(min=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        T = out.size(1)
        mask = (
            torch.arange(T, device=out.device)[None, :]
            < lengths[:, None].to(out.device)
        ).float()

        tweet_rep = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        tweet_rep = tweet_rep.view(B, M, -1)
        tweet_mask = (tweet_len > 0).float().to(tweet_rep.device)

        user_rep = (
            (tweet_rep * tweet_mask.unsqueeze(-1)).sum(dim=1)
            / tweet_mask.sum(dim=1, keepdim=True).clamp(min=1)
        )

        return self.act(self.fc(user_rep))   # (B, D/4)


# ---------------------------------------------------------------------------
#   Feature fusion with Bi-LSTM
# ---------------------------------------------------------------------------
class FeatureFusion(nn.Module):
    
    def __init__(self, sub_dim: int, out_dim: int):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even for Bi-LSTM"
        self.bilstm = nn.LSTM(
            sub_dim,
            out_dim // 2,
            batch_first=True,
            bidirectional=True
        )

    def forward(
        self,
        r_n: torch.Tensor,
        r_c: torch.Tensor,
        r_d: torch.Tensor,
        r_t: torch.Tensor,
    ) -> torch.Tensor:
        seq = torch.stack([r_n, r_c, r_d, r_t], dim=1)   # (B, 4, D/4)
        out, _ = self.bilstm(seq)                        # (B, 4, D)
        x = out.mean(dim=1)                              # (B, D)
        return x


# ---------------------------------------------------------------------------
# Top-level user feature encoder
# ---------------------------------------------------------------------------
class UserFeatureEncoder(nn.Module):
    """
    Shared user feature encoder used by both BotDCGC and CACL.

    Inputs
    ------
    num       : numerical features (static + temporal numerical)
    cat       : categorical features (static + temporal categorical)
    desc_emb  : description embedding
    tweet_emb : tweet text embeddings
    tweet_len : token counts per tweet

    Output
    ------
    x_shared  : shared user representation of shape (N, D)

    This design keeps the rest of the pipelines unchanged:
    BotDCGC, CACL baseline, and HybridModel all consume x_shared only.
    """
    def __init__(
        self,
        num_features: int = 13,
        cat_features: int = 15,
        roberta_dim: int = 768,
        lstm_hidden: int = 128,
        embed_dim: int = 256
    ):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

        sub = embed_dim // 4

        self.num_enc = NumericalEncoder(num_features, sub)
        self.cat_enc = CategoricalEncoder(cat_features, sub)
        self.des_enc = DescriptionEncoder(roberta_dim, sub)
        self.twt_enc = TweetEncoder(roberta_dim, lstm_hidden, sub)
        self.fusion = FeatureFusion(sub, embed_dim)

    def forward(
        self,
        num: torch.Tensor,
        cat: torch.Tensor,
        desc_emb: torch.Tensor,
        tweet_emb: torch.Tensor,
        tweet_len: torch.Tensor,
        return_parts: bool = False
    ):
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