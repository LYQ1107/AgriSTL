import torch
import torch.nn as nn


class ChannelMLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = d_model * expansion
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TimeMixerBlock(nn.Module):
    """
    Simplified TimeMixer block.
    - Time mixing: depthwise 1D convolution over the temporal dimension.
    - Channel mixing: per-timestep MLP over the feature dimension.
    """

    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.0, expansion: int = 4):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.time_mixing = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.channel_mixing = ChannelMLP(d_model=d_model, expansion=expansion, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x_time = x.transpose(1, 2)  # [B, D, T]
        x_time = self.time_mixing(x_time)
        x_time = x_time.transpose(1, 2)  # [B, T, D]
        x = residual + x_time
        x = self.norm1(x)

        residual = x
        x = self.channel_mixing(x)
        x = residual + x
        x = self.norm2(x)
        return x


class TimeMixer_Model(nn.Module):
    r"""
    Simplified TimeMixer implementation.
    - Flatten input from ``(B, T, C, H, W)`` to ``(B, T, C*H*W)`` and project to ``d_model``.
    - Stack TimeMixer blocks for temporal and channel mixing.
    - Project back to the original spatial dimension and keep the final ``pred_len`` steps.

    Args:
        in_shape (tuple): (T, C, H, W)
        seq_len (int): Input sequence length.
        pred_len (int): Prediction sequence length.
        d_model (int): Hidden feature dimension.
        e_layers (int): Number of TimeMixer blocks.
        kernel_size (int): Temporal mixing kernel size.
        dropout (float): Dropout rate.
        expansion (int): Expansion ratio in the channel MLP.
    """

    def __init__(self, in_shape, seq_len=10, pred_len=10, d_model=512, e_layers=4,
                 kernel_size=5, dropout=0.05, expansion=4, **kwargs):
        super().__init__()
        T, C, H, W = in_shape
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = C
        self.spatial_size = C * H * W

        self.input_proj = nn.Linear(self.spatial_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        self.blocks = nn.ModuleList([
            TimeMixerBlock(d_model=d_model, kernel_size=kernel_size, dropout=dropout, expansion=expansion)
            for _ in range(e_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.spatial_size)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x_raw: torch.Tensor, **kwargs) -> torch.Tensor:
        # x_raw: [B, T, C, H, W]
        B, T, C, H, W = x_raw.shape
        assert T == self.seq_len, f"TimeMixer expects seq_len={self.seq_len}, but got {T}"

        x = x_raw.view(B, T, -1)  # [B, T, C*H*W]
        x = self.input_proj(x)    # [B, T, D]
        x = x + self.pos_embedding[:, :T]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.output_proj(x)   # [B, T, C*H*W]
        x = x[:, -self.pred_len:, :]  # [B, pred_len, C*H*W]
        x = x.view(B, self.pred_len, C, H, W)
        return x
