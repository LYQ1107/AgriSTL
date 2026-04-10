import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TimeSformerBlock(nn.Module):
    """Divided space-time attention block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        b, t, n, d = x.shape

        xt = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        xt_norm = self.temporal_norm(xt)
        xt_attn, _ = self.temporal_attn(xt_norm, xt_norm, xt_norm, need_weights=False)
        xt = xt + xt_attn
        x = xt.reshape(b, n, t, d).permute(0, 2, 1, 3)

        xs = x.reshape(b * t, n, d)
        xs_norm = self.spatial_norm(xs)
        xs_attn, _ = self.spatial_attn(xs_norm, xs_norm, xs_norm, need_weights=False)
        xs = xs + xs_attn
        x = xs.reshape(b, t, n, d)

        x = x + self.mlp(self.mlp_norm(x))
        return x


class TimeSformer_Model(nn.Module):
    """A compact TimeSformer for video prediction."""

    def __init__(
        self,
        in_shape,
        seq_len=10,
        pred_len=10,
        patch_size=4,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        t, c, h, w = in_shape
        assert h % patch_size == 0 and w % patch_size == 0, 'H and W must be divisible by patch_size'
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.in_channels = c
        self.height = h
        self.width = w
        self.num_patches_h = h // patch_size
        self.num_patches_w = w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.patch_embed = nn.Conv2d(c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, seq_len, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TimeSformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.head = nn.Linear(embed_dim, c * patch_size * patch_size)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

    def _patchify(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(b, t, self.num_patches, -1)
        return x

    def _unpatchify(self, x):
        b, t, n, _ = x.shape
        x = self.head(x)
        x = x.reshape(
            b, t, self.num_patches_h, self.num_patches_w,
            self.in_channels, self.patch_size, self.patch_size
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        return x.reshape(b, t, self.in_channels, self.height, self.width)

    def forward(self, x_raw, **kwargs):
        x = self._patchify(x_raw)
        x = x + self.pos_embed + self.time_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        x = x.permute(0, 2, 3, 1)
        x = self.time_proj(x)
        x = x.permute(0, 3, 1, 2)
        return self._unpatchify(x)
