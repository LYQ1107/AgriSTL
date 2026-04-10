method = 'Yield3DCNN'

# dataset / model parameters
pre_seq_length = 10
aft_seq_length = 1
in_shape = [10, 4, 16, 16]
out_dim = 1
channel_indices = [0, 6, 7, 8]

# training parameters
lr = 1e-3
batch_size = 1
val_batch_size = 1
epoch = 50
sched = 'onecycle'
