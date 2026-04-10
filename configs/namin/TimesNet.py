method = 'TimesNet'
# model parameters
seq_len = 10  # Input sequence length
pred_len = 10  # Prediction sequence length
modes = 32  # Number of modes for FFT
d_model = 512  # Model dimension (increased for more capacity)
n_heads = 8  # Number of heads
e_layers = 4  # Number of encoder layers (increased)
d_ff = 1024  # Dimension of fcn (increased)
dropout = 0.05  # Dropout rate

# training parameters
lr = 1e-3
batch_size = 16
drop_path = 0
sched = 'onecycle'
