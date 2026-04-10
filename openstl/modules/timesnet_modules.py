import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimesBlock(nn.Module):
    """
    TimesBlock: Optimized version for fast training
    """
    def __init__(self, seq_len=10, pred_len=10, modes=32, d_model=512,
                 n_heads=8, d_ff=512, dropout=0.05):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.modes = modes
        self.d_model = d_model
        self.dropout = dropout

        # Simplified temporal convolution (much faster)
        self.temporal_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model // 4  # Group convolution for speed
        )

        # Multi-head attention (faster than complex 2D operations)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Simplified feed forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, T, N = x.size()
        
        # Temporal convolution
        x_conv = x.permute(0, 2, 1)  # [B, N, T]
        x_conv = self.temporal_conv(x_conv)  # [B, N, T]
        x_conv = x_conv.permute(0, 2, 1)  # [B, T, N]
        
        # Self-attention
        attn_out, _ = self.attention(x_conv, x_conv, x_conv)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class FFT_for_Period(nn.Module):
    """
    FFT for period detection
    """
    def __init__(self, seq_len, modes):
        super(FFT_for_Period, self).__init__()
        self.seq_len = seq_len
        self.modes = modes

    def forward(self, x):
        # [B, T, C]
        B, T, N = x.size()
        
        # Compute FFT
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        
        # Remove DC component
        frequency_list[0] = 0
        
        # Get top frequencies
        num_modes = min(self.modes, len(frequency_list) - 1)
        _, top_list = torch.topk(frequency_list[1:], num_modes)  # Skip DC component
        top_list = top_list + 1  # Adjust for skipped DC component
        top_list = top_list.detach().cpu().numpy()
        
        # Convert to periods
        period = T // top_list
        period = np.maximum(period, 1)  # Ensure minimum period of 1
        
        period_weight = abs(xf).mean(0).mean(-1)[top_list]
        
        return period, period_weight


class Inception_Block_V1(nn.Module):
    """
    Inception block for multi-scale feature extraction
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).sum(-1)
        return res


class Inception_Block_V2(nn.Module):
    """
    Improved Inception block
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).sum(-1)
        return res


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TokenEmbedding(nn.Module):
    """
    Token embedding for time series
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    """
    Data embedding with positional encoding
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
