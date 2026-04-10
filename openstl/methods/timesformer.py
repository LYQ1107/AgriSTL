import torch

from openstl.models import TimeSformer_Model
from .base_method import Base_method


class TimeSformer(Base_method):
    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return TimeSformer_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)[:, :aft_seq_length]
        else:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_pred = self.model(cur_seq)
                pred_y.append(cur_pred)
                cur_seq = cur_pred
            if m != 0:
                pred_y.append(self.model(cur_seq)[:, :m])
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
