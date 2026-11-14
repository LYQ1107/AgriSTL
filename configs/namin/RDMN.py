method = 'RDMN'

# Reverse scheduled sampling
reverse_scheduled_sampling = 1
r_sampling_step_1 = 400
r_sampling_step_2 = 800
r_exp_alpha = 5000

# Scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 800
sampling_start_value = 1.0
sampling_changing_rate = 0.00125

# RDMN architecture used in the AgriSTL experiments
num_hidden = '128,128,128,128'
att_hidden = 64
filter_size = 5
stride = 1
patch_size = 4
layer_norm = 0

# Training
lr = 3e-4
batch_size = 4
val_batch_size = 4
sched = 'cosine'
warmup_epoch = 5
