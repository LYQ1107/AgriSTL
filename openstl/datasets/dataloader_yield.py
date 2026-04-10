import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class YieldDataset(Dataset):
    """Generic crop-yield dataset backed by numpy files.

    Expected sample layout under each split directory:
    - ``*_data.npy`` for input features with shape ``(T, C, H, W)`` or ``(T, H, W, C)``
    - matching ``*_yield.npy`` for regression targets

    The target can be either a scalar or a dense map. To stay compatible with the
    current AgriSTL training pipeline, targets are returned as ``(1, C, H, W)``
    tensors, where scalar targets are reshaped to ``(1, 1, 1, 1)``.
    """

    def __init__(self, samples: List[np.ndarray], targets: List[np.ndarray],
                 data_name: str = 'yield', channel_indices=None):
        super().__init__()
        self.samples = samples
        self.targets = targets
        self.channel_indices = channel_indices
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = np.asarray(self.samples[idx], dtype=np.float32)
        target = np.asarray(self.targets[idx], dtype=np.float32)

        if sample.ndim != 4:
            raise ValueError(f'Yield sample must be 4D, got shape {sample.shape}')

        # Accept either (T, C, H, W) or (T, H, W, C).
        if sample.shape[1] not in (4, 9) and sample.shape[-1] in (4, 9):
            sample = np.transpose(sample, (0, 3, 1, 2))
        if self.channel_indices is not None:
            sample = sample[:, self.channel_indices, :, :]
        sample = np.ascontiguousarray(sample.astype(np.float32, copy=True))

        if target.ndim == 0:
            target = target.reshape(1, 1, 1, 1)
        elif target.ndim == 1:
            target = target.reshape(1, target.shape[0], 1, 1)
        elif target.ndim == 2:
            target = target[None, None, :, :]
        elif target.ndim == 3:
            if target.shape[0] <= 8:
                target = target[None, ...]
            else:
                target = np.transpose(target, (2, 0, 1))[None, ...]
        elif target.ndim == 4:
            pass
        else:
            raise ValueError(f'Unsupported yield target shape {target.shape}')

        target = np.ascontiguousarray(target.astype(np.float32, copy=True))
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def _load_split(split_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not os.path.isdir(split_dir):
        raise ValueError(f'Split directory not found: {split_dir}')

    samples, targets = [], []
    for name in sorted(os.listdir(split_dir)):
        if not name.endswith('_data.npy'):
            continue
        data_path = os.path.join(split_dir, name)
        target_path = os.path.join(split_dir, name.replace('_data.npy', '_yield.npy'))
        if not os.path.exists(target_path):
            raise ValueError(f'Missing target file for {data_path}: expected {target_path}')

        samples.append(np.load(data_path))
        targets.append(np.load(target_path))
    return samples, targets


def load_data(batch_size: int, val_batch_size: int, data_root: str, num_workers: int = 4,
              distributed: bool = False, use_prefetcher: bool = False, drop_last: bool = False, **kwargs):
    root_name = os.path.basename(os.path.normpath(data_root)).lower()
    dataset_root = data_root if root_name.startswith('yield') else os.path.join(data_root, 'yield')
    channel_indices = kwargs.get('channel_indices', [0, 6, 7, 8])

    train_samples, train_targets = _load_split(os.path.join(dataset_root, 'train'))
    val_samples, val_targets = _load_split(os.path.join(dataset_root, 'val'))
    test_samples, test_targets = _load_split(os.path.join(dataset_root, 'test'))

    train_set = YieldDataset(train_samples, train_targets, data_name=root_name or 'yield', channel_indices=channel_indices)
    val_set = YieldDataset(val_samples, val_targets, data_name=root_name or 'yield', channel_indices=channel_indices)
    test_set = YieldDataset(test_samples, test_targets, data_name=root_name or 'yield', channel_indices=channel_indices)

    dataloader_train = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )
    dataloader_vali = create_loader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )
    dataloader_test = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )
    return dataloader_train, dataloader_vali, dataloader_test
