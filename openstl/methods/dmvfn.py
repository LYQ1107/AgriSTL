import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.models import DMVFN_Model
from .base_method import Base_method


def gauss_kernel(size=5, channels=3, device=None, dtype=None):
    kernel = torch.tensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    kernel /= 256.0
    return kernel.repeat(channels, 1, 1, 1)


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    return F.conv2d(img, kernel, groups=img.shape[1])


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x, kernel):
    cc = torch.cat([x, torch.zeros_like(x)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros_like(cc)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * kernel)


def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for _ in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down, kernel)
        pyr.append(current - up)
        current = down
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5):
        super().__init__()
        self.max_levels = max_levels

    def forward(self, pred, target):
        kernel = gauss_kernel(channels=pred.shape[1], device=pred.device, dtype=pred.dtype)
        pyr_pred = laplacian_pyramid(pred, kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(target, kernel, max_levels=self.max_levels)
        return sum(F.l1_loss(a, b) for a, b in zip(pyr_pred, pyr_target))


class DMVFN(Base_method):
    r"""DMVFN autoregressive wrapper for video prediction."""

    def __init__(self, **args):
        super().__init__(**args)
        self.lap_loss = LapLoss()
        self.stage_decay = args.get('stage_decay', 0.8)
        self.use_stage_loss = args.get('use_stage_loss', True)

    def _build_model(self, **args):
        return DMVFN_Model(**args)

    def _rollout(self, batch_x, target_len, return_stage_preds=False):
        context = batch_x
        preds = []
        stage_preds_all = []
        for _ in range(target_len):
            if return_stage_preds:
                stage_preds = self.model(context, return_all_stages=True)
                pred = stage_preds[-1]
                stage_preds_all.append(stage_preds)
            else:
                pred = self.model(context).squeeze(1)
            preds.append(pred)
            context = torch.cat([context[:, 1:], pred.unsqueeze(1)], dim=1)

        preds = torch.stack(preds, dim=1)
        if return_stage_preds:
            return preds, stage_preds_all
        return preds

    def forward(self, batch_x, batch_y=None, **kwargs):
        aft_seq_length = self.hparams.aft_seq_length
        return self._rollout(batch_x, aft_seq_length, return_stage_preds=False)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y, stage_preds_all = self._rollout(batch_x, batch_y.shape[1], return_stage_preds=True)

        loss = self.criterion(pred_y, batch_y)
        if self.use_stage_loss:
            aux_loss = 0.0
            num_stages = len(stage_preds_all[0]) if stage_preds_all else 0
            for step_idx, stage_preds in enumerate(stage_preds_all):
                gt = batch_y[:, step_idx]
                for stage_idx, stage_pred in enumerate(stage_preds):
                    weight = self.stage_decay ** (num_stages - 1 - stage_idx)
                    aux_loss = aux_loss + weight * self.lap_loss(stage_pred, gt)
            loss = loss + aux_loss / max(1, batch_y.shape[1])

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        outputs = {
            'inputs': batch_x.cpu().numpy(),
            'preds': pred_y.cpu().numpy(),
            'trues': batch_y.cpu().numpy(),
        }
        self.test_outputs.append(outputs)
        return outputs
