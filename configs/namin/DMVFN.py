method = 'DMVFN'

# model
scale_list = [4, 4, 4, 2, 2, 2, 1, 1, 1]
num_features = [160, 160, 160, 80, 80, 80, 44, 44, 44]
use_stage_loss = True
stage_decay = 0.8

# training
lr = 1e-4
batch_size = 8
sched = 'onecycle'
