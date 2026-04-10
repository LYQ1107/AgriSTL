method = 'iTransformer'
# model parameters
seq_len = 10  # Input sequence length
pred_len = 10  # Prediction sequence length
d_model = 512  # Model dimension
n_heads = 8  # Number of attention heads
e_layers = 4  # Number of encoder layers (increased for human dataset)
d_ff = 2048  # Dimension of feed forward network
dropout = 0.1  # Dropout rate

# training parameters
lr = 1e-3
batch_size = 16
sched = 'cosine'
warmup_epoch = 0
