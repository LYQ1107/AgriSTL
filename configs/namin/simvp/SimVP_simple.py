method = 'SimVP'
# model - 最小化配置
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'  # 使用更简单的IncepU而不是gSTA
hid_S = 8
hid_T = 32
N_T = 1
N_S = 1
# training
lr = 1e-3
batch_size = 8  # 减小batch size
drop_path = 0
sched = 'onecycle'
