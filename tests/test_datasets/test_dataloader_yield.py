from pathlib import Path

import numpy as np

from openstl.datasets.dataloader_yield import load_data


def _write_split(root: Path, split: str, count: int = 2):
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        sample = np.random.rand(10, 9, 5 + idx, 6 + idx).astype(np.float32)
        target = np.random.rand(5 + idx, 6 + idx).astype(np.float32)
        np.save(split_dir / f'sample_{idx}_data.npy', sample)
        np.save(split_dir / f'sample_{idx}_yield.npy', target)


def test_yield_dataloader(tmp_path):
    dataset_root = tmp_path / 'yield'
    for split in ['train', 'val', 'test']:
        _write_split(dataset_root, split)

    train_loader, val_loader, test_loader = load_data(
        batch_size=1,
        val_batch_size=1,
        data_root=str(tmp_path),
        num_workers=0,
    )

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[0:3] == (1, 10, 4)
    assert batch_x.shape[-2:] in [(5, 6), (6, 7)]
    assert batch_y.shape[0:3] == (1, 1, 1)
    assert batch_y.shape[-2:] in [(5, 6), (6, 7)]
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 2
