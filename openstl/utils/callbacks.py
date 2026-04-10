import json
import shutil
import logging
import os
import os.path as osp
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from .main_utils import check_dir, collect_env, print_log, output_namespace


def _get_metric(trainer, names):
    for name in names:
        if name in trainer.callback_metrics:
            return trainer.callback_metrics.get(name)
    return None


class SetupCallback(Callback):
    def __init__(self, prefix, setup_time, save_dir, ckpt_dir, args, method_info, argv_content=None):
        super().__init__()
        self.prefix = prefix
        self.setup_time = setup_time
        self.save_dir = save_dir
        self.ckpt_dir = ckpt_dir
        self.args = args
        self.config = args.__dict__
        self.argv_content = argv_content
        self.method_info = method_info

    def on_fit_start(self, trainer, pl_module):
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'

        if trainer.global_rank == 0:
            # check dirs
            self.save_dir = check_dir(self.save_dir)
            self.ckpt_dir = check_dir(self.ckpt_dir)
            # setup log
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=logging.INFO,
                filename=osp.join(self.save_dir, '{}_{}.log'.format(self.prefix, self.setup_time)),
                filemode='a', format='%(asctime)s - %(message)s')
            # print env info
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
            sv_param = osp.join(self.save_dir, 'model_param.json')
            with open(sv_param, 'w') as file_obj:
                json.dump(self.config, file_obj)

            print_log(output_namespace(self.args))
            if self.method_info is not None:
                info, flops, fps, dash_line = self.method_info
                print_log('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)


class EpochEndCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        self.avg_train_loss = _get_metric(trainer, ['train_loss_epoch', 'train_loss'])

    def on_validation_epoch_end(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else None
        avg_val_loss = _get_metric(trainer, ['val_loss_epoch', 'val_loss'])

        if hasattr(self, 'avg_train_loss'):
            # 安全处理可能为 None 或 Tensor 的数值
            train_loss_val = self.avg_train_loss
            train_loss_val = train_loss_val.item() if hasattr(train_loss_val, 'item') else train_loss_val
            val_loss_val = avg_val_loss
            val_loss_val = val_loss_val.item() if hasattr(val_loss_val, 'item') else val_loss_val

            lr_str = f"{lr:.7f}" if lr is not None else "N/A"
            train_str = f"{train_loss_val:.7f}" if train_loss_val is not None else "N/A"
            val_str = f"{val_loss_val:.7f}" if val_loss_val is not None else "N/A"
            print_log(f"Epoch {trainer.current_epoch}: Lr: {lr_str} | Train Loss: {train_str} | Vali Loss: {val_str}")


class LossCurveCallback(Callback):
    """Record train & val loss each epoch and dump curve data/figure."""

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        self.history = []
        self._latest_train = None
        self.file_path = osp.join(save_dir, 'loss_history.json')
        self.plot_path = osp.join(save_dir, 'loss_curve.png')

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        metric = _get_metric(trainer, ['train_loss_epoch', 'train_loss'])
        self._latest_train = self._to_float(metric)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_metric = _get_metric(trainer, ['val_loss_epoch', 'val_loss'])
        val_loss = self._to_float(val_metric)

        entry = {
            'epoch': int(trainer.current_epoch),
            'train_loss': self._latest_train,
            'val_loss': val_loss,
        }
        self.history.append(entry)
        self._save_history()
        self._save_plot()

    def _ensure_dir(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def _save_history(self):
        self._ensure_dir()
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    def _save_plot(self):
        if not self.history:
            return
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
        except ImportError:
            return

        epochs = [item['epoch'] for item in self.history]
        train_losses = [item['train_loss'] for item in self.history]
        val_losses = [item['val_loss'] for item in self.history]

        plt.figure(figsize=(8, 3))
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()

        self._ensure_dir()
        plt.savefig(self.plot_path)
        plt.close()

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        if hasattr(value, 'item'):
            return float(value.item())
        return float(value)


class BestCheckpointCallback(ModelCheckpoint):
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))