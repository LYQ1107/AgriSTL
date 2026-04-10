method = 'TimeSformer'

# model parameters
seq_len = 10
pred_len = 10
patch_size = 4
embed_dim = 256
depth = 4
num_heads = 8
mlp_ratio = 4.0
dropout = 0.05

# training parameters
lr = 1e-3
batch_size = 8
drop_path = 0
sched = 'onecycle'
