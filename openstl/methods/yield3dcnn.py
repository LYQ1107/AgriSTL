import os.path as osp

import numpy as np

from openstl.models.yield3dcnn_model import Yield3DCNN_Model
from openstl.utils import check_dir, print_log
from .base_method import Base_method


class Yield3DCNN(Base_method):
    def __init__(self, **args):
        super().__init__(**args)
        self.test_outputs = []

    def _build_model(self, **args):
        return Yield3DCNN_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        return self.model(batch_x)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        diff = pred_y - batch_y
        outputs = {
            'abs_sum': torch_abs_sum(diff),
            'sq_sum': torch_sq_sum(diff),
            'count': int(np.prod(batch_y.shape)),
        }
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        abs_sum = sum(batch['abs_sum'] for batch in self.test_outputs)
        sq_sum = sum(batch['sq_sum'] for batch in self.test_outputs)
        count = sum(batch['count'] for batch in self.test_outputs)

        mae = abs_sum / count
        mse = sq_sum / count
        rmse = np.sqrt(mse)

        eval_res = {'mae': mae, 'mse': mse, 'rmse': rmse}
        eval_log = ', '.join(f'{k}:{v}' for k, v in eval_res.items())

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))
            np.save(osp.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse], dtype=np.float32))

        self.test_outputs.clear()
        return eval_res


def torch_abs_sum(diff):
    return float(diff.detach().abs().sum().cpu().item())


def torch_sq_sum(diff):
    return float(diff.detach().pow(2).sum().cpu().item())
