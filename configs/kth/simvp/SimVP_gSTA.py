method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 16
hid_T = 64
N_T = 6
N_S = 2
# training
lr = 1e-3
drop_path = 0.2
batch_size = 1  # bs = 4 x 4GPUs
sched = 'onecycle'