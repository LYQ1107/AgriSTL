method = 'PredFormer'

model_config = {
    # image h w c
    'height': 128,
    'width': 192,
    'num_channels': 3,
    # video length in and out
    'pre_seq': 8,
    'after_seq': 6,
    # patch size
    'patch_size': 8,
    'dim': 256, 
    'heads': 8,
    'dim_head': 32,
    # dropout
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'scale_dim': 4,
    # depth
    'depth': 1,
    'Ndepth': 6 # For FullAttention-24, for BinaryST, BinaryST, FacST, FacTS-12, for TST,STS-8, for TSST, STTS-6

}
