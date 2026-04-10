import torch
from openstl.models import VMRNN_D_Model, VMRNN_B_Model
from openstl.utils import (reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling)
from .base_method import Base_method

class VMRNN_D(Base_method):
    r"""VMRNN

    Implementation of `VMRNN: A Recurrent Neural Network with Visual Memory for Spatiotemporal
    Predictive Learning`.

    """

    def __init__(self, **args):
        super().__init__(**args)
        self.eta = 1.0

    def _build_model(self, **args):
        depths_downsample = [int(x) for x in self.hparams.depths_downsample.split(',')]
        depths_upsample = [int(x) for x in self.hparams.depths_upsample.split(',')]
        num_heads = [int(x) for x in self.hparams.num_heads.split(',')]
        return VMRNN_D_Model(depths_downsample, depths_upsample, num_heads, self.hparams)

    def forward(self, batch_x, batch_y, **kwargs):
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        img_gen, _ = self.model(test_ims, return_loss=False)
        pred_y = img_gen[:, -self.hparams.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        img_gen, loss = self.model(ims)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

class VMRNN_B(Base_method):
    r"""VMRNN

    Implementation of `VMRNN: A Recurrent Neural Network with Visual Memory for Spatiotemporal
    Predictive Learning`.

    """

    def __init__(self, **args):
        super().__init__(**args)
        self.eta = 1.0

    def _build_model(self, **args):
        return VMRNN_B_Model(self.hparams)

    def forward(self, batch_x, batch_y, **kwargs):
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        img_gen, _ = self.model(test_ims, return_loss=False)
        pred_y = img_gen[:, -self.hparams.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        img_gen, loss = self.model(ims)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss