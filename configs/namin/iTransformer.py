method = 'iTransformer'
# model parameters
seq_len = 10  # Input sequence length
pred_len = 10  # Prediction sequence length
d_model = 256  # Model dimension (fits 8GB GPUs)
n_heads = 4  # Number of attention heads (fits 8GB GPUs)
e_layers = 2  # Number of encoder layers
d_ff = 1024  # Dimension of feed forward network (fits 8GB GPUs)
dropout = 0.1  # Dropout rate

# training parameters
lr = 1e-3
batch_size = 4  # Smaller batch fits in 8GB GPUs
drop_path = 0
sched = 'onecycle'
