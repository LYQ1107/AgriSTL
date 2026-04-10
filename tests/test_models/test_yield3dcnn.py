import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / 'openstl' / 'models' / 'yield3dcnn_model.py'
SPEC = importlib.util.spec_from_file_location('yield3dcnn_model', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
Yield3DCNN_Model = MODULE.Yield3DCNN_Model


def test_yield3dcnn_forward():
    model = Yield3DCNN_Model(in_shape=(10, 4, 5, 6), out_dim=1)
    x = torch.randn(2, 10, 4, 5, 6)
    y = model(x)
    assert y.shape == (2, 1, 1, 5, 6)
