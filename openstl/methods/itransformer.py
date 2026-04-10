import torch
from openstl.models import iTransformer_Model
from .base_method import Base_method


class iTransformer(Base_method):
    r"""iTransformer

    Implementation of `iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    <https://arxiv.org/abs/2310.06625>`_.

    iTransformer is a general framework for time series forecasting based on inverted transformer architecture.
    It applies attention mechanisms and feed-forward networks on the inverted dimensions to capture
    multivariate dependencies and learn non-linear representations.

    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return iTransformer_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        
        # For iTransformer, we need to handle different sequence lengths
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            # For longer prediction, we can use autoregressive prediction
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_pred = self.model(cur_seq)
                pred_y.append(cur_pred)
                # Update input sequence for next prediction
                cur_seq = torch.cat([cur_seq[:, pre_seq_length:], cur_pred], dim=1)
            
            if m != 0:
                cur_pred = self.model(cur_seq)
                pred_y.append(cur_pred[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        
        # Calculate loss
        loss = self.criterion(pred_y, batch_y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        
        # Calculate loss
        loss = self.criterion(pred_y, batch_y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        
        # Calculate loss
        loss = self.criterion(pred_y, batch_y)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
