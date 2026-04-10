import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from openstl.modules.timesnet_modules import TimesBlock, FFT_for_Period, PositionalEncoding


class TimesNet_Model(nn.Module):
    r"""TimesNet Model

    Implementation of `TimesNet: Temporal 2D-Variation Modeling For General Time Series Analysis
    <https://arxiv.org/abs/2210.02186>`_.

    TimesNet is a general framework for time series analysis based on temporal 2D-variation modeling.
    It transforms 1D time series into 2D tensors based on multiple periods derived from FFT,
    then applies 2D kernels to capture temporal variations.

    Args:
        in_shape (tuple): Input shape (T, C, H, W)
        seq_len (int): Input sequence length
        pred_len (int): Prediction sequence length
        modes (int): Number of modes for FFT
        d_model (int): Model dimension
        n_heads (int): Number of heads
        e_layers (int): Number of encoder layers
        d_ff (int): Dimension of fcn
        dropout (float): Dropout rate
        **kwargs: Other arguments
    """

    def __init__(self, in_shape, seq_len=10, pred_len=10, modes=32, d_model=512, 
                 n_heads=8, e_layers=2, d_ff=512, dropout=0.05, **kwargs):
        super(TimesNet_Model, self).__init__()
        T, C, H, W = in_shape
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.modes = modes
        self.d_model = d_model
        
        # Simplified input embedding
        self.input_embedding = nn.Linear(C * H * W, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=1000)
        
        # TimesNet blocks (reduced layers for speed)
        self.timesnet_blocks = nn.ModuleList([
            TimesBlock(
                seq_len=seq_len, 
                pred_len=pred_len, 
                modes=modes, 
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(e_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Simplified output projection
        self.output_projection = nn.Linear(d_model, C * H * W)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        
        # Reshape for time series processing
        x = x_raw.view(B, T, C * H * W)  # [B, T, C*H*W]
        
        # Input embedding
        x = self.input_embedding(x)  # [B, T, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply TimesNet blocks
        for block in self.timesnet_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Output projection
        x = self.output_projection(x)  # [B, T, C*H*W]

        # Use only the required prediction length
        x = x[:, -self.pred_len:, :]  # [B, pred_len, C*H*W]
        
        # Reshape back to original format
        x = x.view(B, self.pred_len, C, H, W)
        
        return x


class TimesNet(nn.Module):
    """
    TimesNet with TimesBlock for time series forecasting.
    """
    def __init__(self, seq_len=96, label_len=48, pred_len=96, modes=32, mode_select='random',
                 version='Fourier', mask_flag=True, factor=1, scale=None, dropout=0.05,
                 output_attention=False, c_out=1, d_model=512, n_heads=8, e_layers=2,
                 d_layers=1, d_ff=512, activation='gelu', distil=True):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(
            seq_len=seq_len, pred_len=pred_len, modes=modes, mode_select=mode_select,
            version=version, mask_flag=mask_flag, factor=factor, scale=scale,
            dropout=dropout, output_attention=output_attention, c_out=c_out,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, activation=activation
        ) for _ in range(e_layers)])
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        if distil:
            self.projection = nn.Linear(d_model, c_out, bias=True)
        else:
            self.projection = nn.Linear(d_model, c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class DataEmbedding_wo_pos(nn.Module):
    """
    Data embedding without positional encoding
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Token embedding
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
