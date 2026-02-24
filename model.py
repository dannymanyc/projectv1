# model.py
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalePatchBranch(nn.Module):
    """
    One branch = optional downsampling + patch embedding via Conv1D.

    Input:  x -> (B, T, F)
    Output: tokens -> (B, N_patches, D)
    """
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        d_model: int,
        downsample: int = 1,
        patch_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert downsample >= 1, "downsample must be >= 1"
        assert patch_size >= 1, "patch_size must be >= 1"

        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.downsample = downsample
        self.patch_size = patch_size

        reduced_len = seq_len // downsample
        if reduced_len < patch_size:
            raise ValueError(
                f"ScalePatchBranch invalid: reduced_len={reduced_len} < patch_size={patch_size}. "
                f"Increase seq_len or reduce downsample/patch_size."
            )

        # Number of patches produced by Conv1d(kernel=stride=patch_size)
        self.n_patches = reduced_len // patch_size

        # Conv1d patch embed: channels = features, time = sequence
        self.patch_embed = nn.Conv1d(
            in_channels=num_features,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.token_norm = nn.LayerNorm(d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.scale_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.scale_embedding, mean=0.0, std=0.02)

    def _downsample_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns: (B, T_reduced, F)
        """
        if self.downsample == 1:
            return x

        # avg-pool along time dimension
        # x_t: (B, F, T)
        x_t = x.transpose(1, 2)
        x_t = F.avg_pool1d(
            x_t,
            kernel_size=self.downsample,
            stride=self.downsample,
            ceil_mode=False,
        )
        return x_t.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns tokens: (B, N, D)
        """
        x = self._downsample_time(x)  # (B, T', F)
        x = x.transpose(1, 2)         # (B, F, T')
        x = self.patch_embed(x)       # (B, D, N)
        x = x.transpose(1, 2)         # (B, N, D)
        x = self.token_norm(x)
        x = x + self.pos_embedding + self.scale_embedding
        x = self.dropout(x)
        return x


class AttentionPool(nn.Module):
    """
    Learnable attention pooling over tokens.
    Input:  (B, N, D)
    Output: (B, D)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        # scores: (B, N)
        scores = torch.einsum("bnd,d->bn", x, self.query) * self.scale
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bn,bnd->bd", weights, x)
        return pooled


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchMultiScaleTimeSeriesTransformer(nn.Module):
    """
    Upgraded transformer for time-series classification/regression:
    - multi-scale branches (1m / 5m / 15m style via downsampling)
    - patch embedding (reduces token count)
    - encoder-only transformer
    - CLS token + attention pooling + mean pooling readout
    - optional multi-task heads (magnitude / volatility)

    Inputs:
        x: (B, seq_len, num_features)

    Outputs:
        dict with:
            logits: (B, num_classes)
            magnitude: (B,)         [optional]
            volatility: (B,)        [optional]
    """
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        downsample_factors: Optional[List[int]] = None,
        patch_sizes: Optional[List[int]] = None,
        dim_feedforward_mult: int = 4,
        head_hidden_mult: int = 2,
        use_magnitude_head: bool = False,
        use_volatility_head: bool = False,
    ):
        super().__init__()

        if downsample_factors is None:
            downsample_factors = [1, 5, 15]
        if patch_sizes is None:
            patch_sizes = [5, 2, 1]  # after downsampling: ~5m, ~10m, ~15m chunks

        if len(downsample_factors) != len(patch_sizes):
            raise ValueError("downsample_factors and patch_sizes must have same length")

        self.num_features = num_features
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_magnitude_head = use_magnitude_head
        self.use_volatility_head = use_volatility_head

        # Normalize raw features per time step before patching (helps stability)
        self.input_norm = nn.LayerNorm(num_features)

        # Multi-scale branches
        self.branches = nn.ModuleList([
            ScalePatchBranch(
                num_features=num_features,
                seq_len=seq_len,
                d_model=d_model,
                downsample=ds,
                patch_size=ps,
                dropout=dropout,
            )
            for ds, ps in zip(downsample_factors, patch_sizes)
        ])

        total_tokens = sum(branch.n_patches for branch in self.branches)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * dim_feedforward_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.post_norm = nn.LayerNorm(d_model)
        self.attn_pool = AttentionPool(d_model)

        # Readout combines:
        # CLS token + attention pooled tokens + mean pooled tokens
        readout_dim = d_model * 3
        self.readout_norm = nn.LayerNorm(readout_dim)
        self.readout_dropout = nn.Dropout(dropout)

        head_hidden = d_model * head_hidden_mult
        self.classifier = MLPHead(readout_dim, head_hidden, num_classes, dropout=dropout)

        if self.use_magnitude_head:
            self.magnitude_head = MLPHead(readout_dim, head_hidden, 1, dropout=dropout)
        else:
            self.magnitude_head = None

        if self.use_volatility_head:
            self.volatility_head = MLPHead(readout_dim, head_hidden, 1, dropout=dropout)
        else:
            self.volatility_head = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # MLPHead already initializes, but this is safe
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, T, F)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B,T,F), got {tuple(x.shape)}")

        x = self.input_norm(x)

        # Build tokens from all scales
        branch_tokens = [branch(x) for branch in self.branches]  # list[(B, N_i, D)]
        tokens = torch.cat(branch_tokens, dim=1)                 # (B, N_total, D)

        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, D)
        x_tok = torch.cat([cls, tokens], dim=1)                  # (B, 1+N, D)

        # Encode
        x_tok = self.transformer_encoder(x_tok)                  # (B, 1+N, D)
        x_tok = self.post_norm(x_tok)

        cls_out = x_tok[:, 0, :]                                 # (B, D)
        token_out = x_tok[:, 1:, :]                              # (B, N, D)

        attn_pooled = self.attn_pool(token_out)                  # (B, D)
        mean_pooled = token_out.mean(dim=1)                      # (B, D)

        readout = torch.cat([cls_out, attn_pooled, mean_pooled], dim=-1)  # (B, 3D)
        readout = self.readout_norm(readout)
        readout = self.readout_dropout(readout)

        out: Dict[str, torch.Tensor] = {
            "logits": self.classifier(readout)
        }

        if self.magnitude_head is not None:
            out["magnitude"] = self.magnitude_head(readout).squeeze(-1)

        if self.volatility_head is not None:
            out["volatility"] = self.volatility_head(readout).squeeze(-1)

        return out


# Optional flash-attn check (placeholder notice)
try:
    import flash_attn  # noqa: F401
    FLASH_AVAILABLE = True
except Exception:
    FLASH_AVAILABLE = False