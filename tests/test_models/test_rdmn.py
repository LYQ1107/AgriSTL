from types import SimpleNamespace

import torch

from openstl.methods import method_maps
from openstl.models.rdmn_model import RDMNModel


def test_rdmn_registry_and_training_forward():
    assert method_maps['rdmn'] is method_maps['gmg_mim_sgfm']

    config = SimpleNamespace(
        in_shape=[2, 3, 16, 16],
        pre_seq_length=2,
        aft_seq_length=2,
        patch_size=4,
        filter_size=3,
        stride=1,
        layer_norm=0,
    )
    model = RDMNModel(2, [4, 4], 2, config)
    frames = torch.rand(1, 4, 4, 4, 48)
    mask = torch.zeros(1, 1, 4, 4, 48)

    prediction, loss = model(frames, mask)

    assert prediction.shape == (1, 3, 4, 4, 48)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
