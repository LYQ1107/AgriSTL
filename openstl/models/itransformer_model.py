import torch
import torch.nn as nn
from openstl.modules.itransformer_modules import (
    InvertedEmbedding, InvertedTransformerBlock, ProjectionLayer
)


class iTransformer_Model(nn.Module):
    r"""iTransformer Model

    Implementation of `iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    <https://arxiv.org/abs/2310.06625>`_.

    iTransformer is a general framework for time series forecasting based on inverted transformer architecture.
    It applies attention mechanisms and feed-forward networks on the inverted dimensions to capture
    multivariate dependencies and learn non-linear representations.

    Args:
        in_shape (tuple): Input shape (T, C, H, W)
        seq_len (int): Input sequence length
        pred_len (int): Prediction sequence length
        d_model (int): Model dimension
        n_heads (int): Number of heads
        e_layers (int): Number of encoder layers
        d_ff (int): Dimension of feed forward network
        dropout (float): Dropout rate
        **kwargs: Other arguments
    """

    def __init__(self, in_shape, seq_len=10, pred_len=10, d_model=512, 
                 n_heads=8, e_layers=2, d_ff=2048, dropout=0.1, **kwargs):
        super(iTransformer_Model, self).__init__()
        T, C, H, W = in_shape
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Calculate the number of variables (flattened spatial dimensions)
        self.num_variables = C * H * W
        
        # Inverted embedding
        self.embedding = InvertedEmbedding(
            c_in=seq_len,  # Each variable has seq_len time steps
            d_model=d_model,
            dropout=dropout
        )
        
        # Inverted transformer blocks
        self.transformer_blocks = nn.ModuleList([
            InvertedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(e_layers)
        ])
        
        # Projection layer
        self.projection = ProjectionLayer(
            d_model=d_model,
            c_out=pred_len,  # Output prediction length
            dropout=dropout
        )

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        
        # Reshape for inverted processing: [B, T, C*H*W] -> [B, C*H*W, T]
        x = x_raw.view(B, T, C * H * W)  # [B, T, N] where N = C*H*W
        x = x.transpose(1, 2)  # [B, N, T] - inverted for iTransformer
        
        # Apply embedding to each variable with memory optimization
        x_embedded = []
        # Process variables in chunks to reduce memory usage
        chunk_size = min(1024, self.num_variables)  # Process in chunks of 1024 or less
        for i in range(0, self.num_variables, chunk_size):
            end_idx = min(i + chunk_size, self.num_variables)
            chunk_data = x[:, i:end_idx, :]  # [B, chunk_size, T]
            
            chunk_embedded = []
            for j in range(chunk_data.size(1)):
                var_data = chunk_data[:, j, :].unsqueeze(1)  # [B, 1, T]
                var_embedded = self.embedding(var_data)  # [B, 1, T, d_model]
                chunk_embedded.append(var_embedded)
            
            chunk_embedded = torch.cat(chunk_embedded, dim=1)  # [B, chunk_size, T, d_model]
            x_embedded.append(chunk_embedded)
        
        x = torch.cat(x_embedded, dim=1)  # [B, N, T, d_model]
        
        # Apply inverted transformer blocks
        for block in self.transformer_blocks:
            x = block(x)  # [B, N, T, d_model]
        
        # Project to prediction length
        x = self.projection(x)  # [B, N, T, pred_len]
        
        # Reshape back to original format: [B, N, T, pred_len] -> [B, pred_len, C, H, W]
        # Take the last time step for prediction
        x = x[:, :, -1, :]  # [B, N, pred_len] - take last time step
        x = x.transpose(1, 2)  # [B, pred_len, N]
        x = x.contiguous()  # Ensure tensor is contiguous
        x = x.reshape(B, self.pred_len, C, H, W)  # Use reshape instead of view
        
        return x


class iTransformer(nn.Module):
    """
    iTransformer with inverted transformer blocks for time series forecasting.
    """
    def __init__(self, seq_len=96, label_len=48, pred_len=96, d_model=512, 
                 n_heads=8, e_layers=2, d_ff=2048, dropout=0.1, c_out=1):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.c_out = c_out
        
        # Inverted embedding
        self.embedding = InvertedEmbedding(
            c_in=seq_len,
            d_model=d_model,
            dropout=dropout
        )
        
        # Inverted transformer blocks
        self.transformer_blocks = nn.ModuleList([
            InvertedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(e_layers)
        ])
        
        # Projection layer
        self.projection = ProjectionLayer(
            d_model=d_model,
            c_out=c_out,
            dropout=dropout
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Inverted processing: [B, L, N] -> [B, N, L]
        x_enc = x_enc.transpose(1, 2)  # [B, N, L]
        
        # Add embedding dimension
        B, N, L = x_enc.shape
        x_enc = x_enc.unsqueeze(-1).expand(B, N, L, self.d_model)  # [B, N, L, d_model]
        x_enc = x_enc.transpose(1, 2)  # [B, L, N, d_model]
        
        # Apply embedding
        x_enc = self.embedding(x_enc)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x_enc = block(x_enc)
        
        # Project to prediction
        dec_out = self.projection(x_enc)  # [B, L, N, pred_len]
        
        # Reshape for output
        dec_out = dec_out.transpose(1, 3)  # [B, pred_len, N, L]
        dec_out = dec_out[:, :, :, 0]  # [B, pred_len, N]
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
