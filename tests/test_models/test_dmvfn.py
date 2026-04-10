import torch

from openstl.models import DMVFN_Model


def test_dmvfn_forward_shape():
    model = DMVFN_Model(in_shape=(10, 3, 32, 32))
    x = torch.rand(2, 10, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 1, 3, 32, 32)


def test_dmvfn_stage_outputs():
    model = DMVFN_Model(in_shape=(10, 3, 32, 32))
    x = torch.rand(1, 10, 3, 32, 32)
    stage_preds = model(x, return_all_stages=True)
    assert len(stage_preds) == 9
    assert all(pred.shape == (1, 3, 32, 32) for pred in stage_preds)
