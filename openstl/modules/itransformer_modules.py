import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class InvertedEmbedding(nn.Module):
    """
    Inverted embedding for iTransformer
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super(InvertedEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [B, 1, T] -> [B, 1, T, d_model]
        # x: [B, 1, T] -> [B, T, 1] -> [B, T, d_model] -> [B, 1, T, d_model]
        x = x.transpose(1, 2)  # [B, T, 1]
        x = x.squeeze(-1)  # [B, T] - remove the last dimension
        x = self.value_embedding(x)  # [B, T, d_model]
        x = x.unsqueeze(1)  # [B, 1, T, d_model]
        x = self.position_embedding(x)
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for iTransformer
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)  # [1, 1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, 1, T, d_model]
        return x + self.pe[:, :, :x.size(2), :]


class InvertedAttention(nn.Module):
    """
    Inverted attention mechanism for iTransformer
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(InvertedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, N, T, d_model]
        B, N, T, d_model = x.size()
        
        # Apply attention across variables (N dimension) with chunked processing
        x_flat = x.view(B * T, N, d_model)  # [B*T, N, d_model]
        
        # Multi-head attention with chunked processing to reduce memory usage
        Q = self.w_q(x_flat).view(B * T, N, self.n_heads, self.d_k).transpose(1, 2)  # [B*T, n_heads, N, d_k]
        K = self.w_k(x_flat).view(B * T, N, self.n_heads, self.d_k).transpose(1, 2)  # [B*T, n_heads, N, d_k]
        V = self.w_v(x_flat).view(B * T, N, self.n_heads, self.d_k).transpose(1, 2)  # [B*T, n_heads, N, d_k]
        
        # Ultra-memory-efficient attention computation
        # Use extremely small chunks and process one time step at a time
        if N > 256:  # Use chunked processing for large N
            chunk_size = min(32, N)  # Extremely small chunks
            attn_output = torch.zeros_like(V)  # [B*T, n_heads, N, d_k]
            
            # Process one time step at a time to minimize memory usage
            for t in range(B * T):
                Q_t = Q[t:t+1]  # [1, n_heads, N, d_k]
                K_t = K[t:t+1]  # [1, n_heads, N, d_k]
                V_t = V[t:t+1]  # [1, n_heads, N, d_k]
                
                for i in range(0, N, chunk_size):
                    end_i = min(i + chunk_size, N)
                    
                    # Compute attention for this chunk
                    Q_chunk = Q_t[:, :, i:end_i, :]  # [1, n_heads, chunk_size, d_k]
                    
                    # Compute attention scores for this chunk against all keys
                    scores_chunk = torch.matmul(Q_chunk, K_t.transpose(-2, -1)) / math.sqrt(self.d_k)  # [1, n_heads, chunk_size, N]
                    attn_weights_chunk = F.softmax(scores_chunk, dim=-1)
                    attn_weights_chunk = self.dropout(attn_weights_chunk)
                    
                    # Apply attention to values
                    attn_output_chunk = torch.matmul(attn_weights_chunk, V_t)  # [1, n_heads, chunk_size, d_k]
                    attn_output[t, :, i:end_i, :] = attn_output_chunk
                    
                    # Clear intermediate tensors to free memory
                    del Q_chunk, scores_chunk, attn_weights_chunk, attn_output_chunk
                    torch.cuda.empty_cache()
        else:
            # Standard attention for smaller N
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B*T, n_heads, N, N]
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, V)  # [B*T, n_heads, N, d_k]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B * T, N, d_model)  # [B*T, N, d_model]
        attn_output = self.w_o(attn_output)  # [B*T, N, d_model]
        
        # Reshape back
        attn_output = attn_output.view(B, N, T, d_model)  # [B, N, T, d_model]
        
        # Residual connection and layer norm
        output = self.layer_norm(x + attn_output)
        return output


class InvertedFeedForward(nn.Module):
    """
    Inverted feed forward network for iTransformer
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(InvertedFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [B, N, T, d_model]
        B, N, T, d_model = x.size()
        
        # Apply FFN across variables (N dimension)
        x_flat = x.view(B * T, N, d_model)  # [B*T, N, d_model]
        
        # Feed forward
        ff_output = self.linear1(x_flat)  # [B*T, N, d_ff]
        ff_output = self.activation(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)  # [B*T, N, d_model]
        ff_output = self.dropout(ff_output)
        
        # Reshape back
        ff_output = ff_output.view(B, N, T, d_model)  # [B, N, T, d_model]
        
        # Residual connection and layer norm
        output = self.layer_norm(x + ff_output)
        return output


class InvertedTransformerBlock(nn.Module):
    """
    Inverted Transformer block for iTransformer
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(InvertedTransformerBlock, self).__init__()
        self.attention = InvertedAttention(d_model, n_heads, dropout)
        self.feed_forward = InvertedFeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        # x: [B, L, N, d_model]
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class ProjectionLayer(nn.Module):
    """
    Projection layer for iTransformer
    """
    def __init__(self, d_model, c_out, dropout=0.1):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(d_model, c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, T, d_model]
        B, N, T, d_model = x.size()
        
        # Apply projection across variables
        x_flat = x.view(B * T, N, d_model)  # [B*T, N, d_model]
        x_proj = self.projection(x_flat)  # [B*T, N, c_out]
        x_proj = self.dropout(x_proj)
        
        # Reshape back
        x_proj = x_proj.view(B, N, T, -1)  # [B, N, T, c_out]
        return x_proj
