import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / 'openstl' / 'models' / 'timesformer_model.py'
SPEC = importlib.util.spec_from_file_location('timesformer_model', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
TimeSformer_Model = MODULE.TimeSformer_Model


def test_timesformer_forward():
    model = TimeSformer_Model(
        in_shape=(10, 3, 64, 64),
        seq_len=10,
        pred_len=10,
        patch_size=4,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )
    x = torch.randn(2, 10, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 10, 3, 64, 64)
