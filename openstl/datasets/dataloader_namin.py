# dataloader_namin.py (modified)
import os
import re
import random
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


def natural_sort_key(s: str):
    parts = re.findall(r'\d+|\D+', s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


class NaminDataset(Dataset):
    """
    Dataset for namin plant sequences.

    datas: numpy array (num_frames_total, H, W, C)
    indices: list of start indices for sequences
    """
    def __init__(self, datas: np.ndarray, indices: List[int],
                 pre_seq_length: int, aft_seq_length: int, use_augment: bool = False,
                 data_name: str = 'namin', expected_channels: int = 3):
        super().__init__()
        # datas expected shape: (num_frames_total, H, W, C)
        # validate channels and optionally adapt
        if datas.size == 0:
            # keep shape consistent for empty dataset
            self.datas = datas.transpose(0, 3, 1, 2) if datas.ndim == 4 else datas
        else:
            if datas.ndim != 4:
                raise ValueError(f"datas expected 4D (F,H,W,C), got shape {datas.shape}")
            c = datas.shape[3]
            if c != expected_channels:
                # try to adapt: common cases: convert RGB->Gray, Gray->RGB, repeat channels
                if c == 3 and expected_channels == 1:
                    # rgb -> gray
                    gray = cv2.cvtColor(datas[0], cv2.COLOR_RGB2GRAY)  # dummy to get dtype; we'll do vectorized below
                    # vectorized conversion
                    data_gray = (0.2989 * datas[..., 0] + 0.5870 * datas[..., 1] + 0.1140 * datas[..., 2]).astype(datas.dtype)
                    data_gray = data_gray[..., None]  # (F, H, W, 1)
                    datas = data_gray
                elif c == 1 and expected_channels == 3:
                    # 1 -> 3
                    datas = np.repeat(datas, 3, axis=3)
                else:
                    # general strategy: repeat channels or truncate
                    if expected_channels > c:
                        reps = (expected_channels + c - 1) // c
                        datas = np.tile(datas, (1, 1, 1, reps))[:, :, :, :expected_channels]
                    else:
                        datas = datas[:, :, :, :expected_channels]
                print(f"[NaminDataset] Adapted data channels from {c} -> {expected_channels}")

            # convert to (num_frames_total, C, H, W)
            self.datas = datas.transpose(0, 3, 1, 2)  # (F, C, H, W)

        self.indices = indices
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.use_augment = use_augment

        # --- 新增字段（修复 AttributeError） ---
        # 与 KTHDataset 保持一致，方便上层代码访问
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def _augment_seq(self, imgs: torch.Tensor, crop_scale: float = 0.95):
        # imgs: (T, C, H, W)
        t, c, h, w = imgs.shape
        # Upscale each frame slightly then random crop back to original size
        frames = []
        for frame in imgs:
            f = frame.unsqueeze(0)  # (1, C, H, W)
            ih = int(round(h / crop_scale))
            iw = int(round(w / crop_scale))
            f_up = F.interpolate(f, size=(ih, iw), mode='bilinear', align_corners=False)
            frames.append(f_up.squeeze(0))
        imgs_up = torch.stack(frames, dim=0)  # (T, C, H_up, W_up)
        _, _, ih, iw = imgs_up.shape
        x = random.randint(0, max(0, ih - h))
        y = random.randint(0, max(0, iw - w))
        imgs_crop = imgs_up[:, :, x:x+h, y:y+w]
        if random.randint(0, 1):
            imgs_crop = torch.flip(imgs_crop, dims=(3,))
        return imgs_crop

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        start = self.indices[i]
        end1 = start + self.pre_seq_length
        end2 = end1 + self.aft_seq_length
        data = torch.from_numpy(self.datas[start:end1].astype(np.float32))  # (pre, C, H, W)
        labels = torch.from_numpy(self.datas[end1:end2].astype(np.float32))  # (aft, C, H, W)
        if self.use_augment:
            seq = torch.cat([data, labels], dim=0)
            seq = self._augment_seq(seq, crop_scale=0.95)
            data = seq[:self.pre_seq_length]
            labels = seq[self.pre_seq_length:]
        return data, labels



class InputHandle:
    """Simple input handler similar to KTH version for compatibility."""
    def __init__(self, datas: np.ndarray, indices: List[int], input_param: dict):
        self.name = input_param.get('name', 'namin')
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas  # (F, H, W, C)
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        return self.current_position + self.minibatch_size >= self.total()

    def get_batch(self):
        if self.no_batch_left():
            print('No batch left. Call begin() to restart.')
            return None
        b = self.minibatch_size
        seq_len = self.current_input_length
        W = self.image_width
        input_batch = np.zeros((b, seq_len, W, W, 3)).astype(self.input_data_type)
        for i in range(b):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + seq_len
            data_slice = self.datas[begin:end, :, :, :]
            input_batch[i, :seq_len, :, :, :] = data_slice
        return input_batch


class DataProcess:
    """
    Data loader for namin dataset.
    Expects structure:
      {paths}/train/<plant_folder>/*.jpg
      {paths}/val/<plant_folder>/*.jpg
      {paths}/test/<plant_folder>/*.jpg
    Each plant_folder contains the time-series images (e.g., 22 images).
    """
    def __init__(self, input_param: dict):
        self.paths = input_param['paths']
        self.image_width = input_param['image_width']  # will be 128
        self.input_param = input_param
        self.seq_len = input_param['seq_length']
        # optional expected channels (for safety)
        self.expected_channels = input_param.get('channels', 3)

    def _list_images(self, folder: str) -> List[str]:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
                 and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort(key=natural_sort_key)
        return files

    def load_data(self, base_path: str, mode: str = 'train') -> Tuple[np.ndarray, List[int]]:
        """
        Read images under base_path/mode. Each subfolder is one plant sequence.
        Returns:
            data: numpy array shape (total_frames, H, W, C), float32 in [0,1]
            indices: list of starting indices for sequences of length seq_len
        """
        assert mode in ['train', 'val', 'test']
        split_dir = os.path.join(base_path, mode)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")

        frames_list = []
        indices = []
        total_frames_so_far = 0

        folder_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))],
                              key=natural_sort_key)

        for folder_name in folder_names:
            folder_path = os.path.join(split_dir, folder_name)
            imgs = self._list_images(folder_path)
            if len(imgs) == 0:
                continue
            # read and resize frames
            for img_name in imgs:
                img_path = os.path.join(folder_path, img_name)
                with Image.open(img_path) as im:
                    im = im.convert('RGB')
                    arr = np.array(im)  # (H_orig, W_orig, 3)
                # resize to image_width x image_width
                arr_resized = cv2.resize(arr, (self.image_width, self.image_width))
                frames_list.append(arr_resized)
            # add sliding-window start indices for this folder
            folder_frame_count = len(imgs)
            if folder_frame_count >= self.seq_len:
                for s in range(0, folder_frame_count - self.seq_len + 1):
                    indices.append(total_frames_so_far + s)
            total_frames_so_far += folder_frame_count

        if len(frames_list) == 0:
            data = np.zeros((0, self.image_width, self.image_width, self.expected_channels), dtype=np.float32)
            return data, []

        frames_np = np.asarray(frames_list)  # (F, H, W, 3)
        # normalize to [0,1]
        data = frames_np.astype(np.float32) / 255.0

        # ensure channel count matches expected_channels
        if data.shape[3] != self.expected_channels:
            print(f"[DataProcess] data channels {data.shape[3]} != expected {self.expected_channels}. Adapting...")
            if data.shape[3] == 3 and self.expected_channels == 1:
                data = (0.2989 * data[..., 0] + 0.5870 * data[..., 1] + 0.1140 * data[..., 2])[..., None]
            elif data.shape[3] == 1 and self.expected_channels == 3:
                data = np.repeat(data, 3, axis=3)
            else:
                # repeat/truncate strategy
                c = data.shape[3]
                if self.expected_channels > c:
                    reps = (self.expected_channels + c - 1) // c
                    data = np.tile(data, (1, 1, 1, reps))[:, :, :, :self.expected_channels]
                else:
                    data = data[..., :self.expected_channels]

        return data, indices

    def get_train_input_handle(self):
        data, indices = self.load_data(self.paths, mode='train')
        return InputHandle(data, indices, self.input_param)

    def get_test_input_handle(self):
        data, indices = self.load_data(self.paths, mode='test')
        return InputHandle(data, indices, self.input_param)

    def get_val_input_handle(self):
        data, indices = self.load_data(self.paths, mode='val')
        return InputHandle(data, indices, self.input_param)


def load_data(batch_size: int, val_batch_size: int, data_root: str, num_workers: int = 4,
              pre_seq_length: int = 10, aft_seq_length: int = 12, in_shape: list = [22, 3, 128, 128],
              distributed: bool = False, use_augment: bool = False, use_prefetcher: bool = False,
              drop_last: bool = False):
    """
    Build dataloaders for namin dataset.
    data_root: folder that contains 'namin' subfolder, or directly the 'namin' folder.
    """

    img_width = in_shape[-1] if in_shape is not None else 128
    channels = in_shape[1] if in_shape is not None and len(in_shape) > 1 else 3
    input_param = {
        'paths': os.path.join(data_root, 'namin') if os.path.basename(data_root) != 'namin' else data_root,
        'image_width': img_width,  # will be 128
        'minibatch_size': batch_size,
        'seq_length': (pre_seq_length + aft_seq_length),
        'input_data_type': 'float32',
        'name': 'namin',
        'channels': channels
    }

    dp = DataProcess(input_param)
    train_handle = dp.get_train_input_handle()
    val_handle = dp.get_val_input_handle()
    test_handle = dp.get_test_input_handle()

    train_set = NaminDataset(train_handle.datas, train_handle.indices,
                             pre_seq_length, aft_seq_length, use_augment=use_augment,
                             expected_channels=channels)
    val_set = NaminDataset(val_handle.datas, val_handle.indices,
                           pre_seq_length, aft_seq_length, use_augment=False,
                           expected_channels=channels)
    test_set = NaminDataset(test_handle.datas, test_handle.indices,
                            pre_seq_length, aft_seq_length, use_augment=False,
                            expected_channels=channels)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(val_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


# -------------------------
# Utility functions for debugging/loading checkpoints safely
# -------------------------
def print_model_bn_info(model: nn.Module):
    """Print BatchNorm layers and their num_features for quick debug."""
    print("=== Model BatchNorm layers info ===")
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            print(f"{name}: {type(module).__name__}, num_features = {getattr(module, 'num_features', None)}")


def print_state_dict_shapes(state_dict: dict):
    """Print shapes of tensors in a state_dict for quick inspection."""
    print("=== state_dict shapes ===")
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape))


def filter_and_load_checkpoint(model: nn.Module, ckpt_path: str, device='cpu', remove_bn_stats=True):
    """
    Load checkpoint but only load parameters whose shapes match the model.
    If remove_bn_stats is True, running_mean/running_var entries that don't match model will be dropped.
    Returns: (loaded_keys, skipped_keys)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    # support cases: ckpt may be a dict with 'state_dict' or direct state dict
    if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    model_state = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k not in model_state:
            skipped.append((k, 'missing in model'))
            continue
        if isinstance(v, torch.Tensor) and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            # if it's BN stats and remove_bn_stats True, skip silently
            if remove_bn_stats and ('running_mean' in k or 'running_var' in k):
                skipped.append((k, f"bn_stat_shape_mismatch {getattr(v,'shape',None)} != {model_state[k].shape}"))
                continue
            skipped.append((k, f"shape_mismatch {getattr(v,'shape',None)} != {model_state[k].shape}"))

    # load filtered params
    model.load_state_dict(filtered, strict=False)
    print(f"[filter_and_load_checkpoint] loaded {len(filtered)} keys, skipped {len(skipped)} keys.")
    if len(skipped) > 0:
        print("Some skipped keys (name, reason):")
        for s in skipped[:40]:
            print(" ", s)
    return list(filtered.keys()), skipped


if __name__ == '__main__':
    # quick test (adjust data_root to your project layout)
    data_root = '../../data'  # expects ../../data/namin/train, ../../data/namin/val, ../../data/namin/test
    train_loader, val_loader, test_loader = load_data(batch_size=8,
                                                      val_batch_size=4,
                                                      data_root=data_root,
                                                      num_workers=4,
                                                      pre_seq_length=10,
                                                      aft_seq_length=12,
                                                      in_shape=[22, 3, 128, 128],
                                                      use_augment=False)
    print("Train batches:", len(train_loader) if train_loader is not None else 0)
    for x, y in train_loader:
        print("batch x:", x.shape, "batch y:", y.shape)
        break
    for x, y in test_loader:
        print("test x:", x.shape, "test y:", y.shape)
        break
