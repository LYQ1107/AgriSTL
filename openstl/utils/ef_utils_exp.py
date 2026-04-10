import os
import sys
import logging
import datetime
import numpy as np


def create_exp_dir(args):
    """
    创建实验目录，结构如下：
    outputs/
      └── {exp_name}_{timestamp}/
          ├── checkpoints/  (存放 .pth)
          ├── logs/         (存放 .log)
          └── saved/        (存放 .npy 结果)
    """
    # 1. 生成带时间戳的实验名
    timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
    exp_name = f"{args.ex_name}_{timestamp}"

    # 2. 根目录
    base_dir = os.path.join("./outputs", exp_name)

    # 3. 子目录
    dirs = {
        'root': base_dir,
        'ckpt': os.path.join(base_dir, 'checkpoints'),
        'logs': os.path.join(base_dir, 'logs'),
        'saved': os.path.join(base_dir, 'saved'),  # 专门放 npy
    }

    for k, v in dirs.items():
        os.makedirs(v, exist_ok=True)

    print(f"📁 实验目录已创建: {base_dir}")
    return dirs


def setup_logger(log_dir):
    """配置 Logger，同时输出到控制台和文件"""
    log_file = os.path.join(log_dir, 'train.log')

    # 获取 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除之前的 handler (防止重复打印)
    logger.handlers = []

    # 1. File Handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    # 2. Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)

    return logger


def save_agristl_results(save_dir, inputs, trues, preds, metrics):
    """
    保存符合 AgriSTL 格式的 .npy 文件
    inputs: (N, T_in, C, H, W)
    trues:  (N, T_out, C, H, W)
    preds:  (N, T_out, C, H, W)
    metrics: dict
    """
    print(f"💾 正在保存 NPY 文件到 {save_dir} ...")
    np.save(os.path.join(save_dir, 'inputs.npy'), inputs)
    np.save(os.path.join(save_dir, 'trues.npy'), trues)
    np.save(os.path.join(save_dir, 'preds.npy'), preds)

    # AgriSTL 的 metrics 通常也是存成 npy 或者简单的 dict
    np.save(os.path.join(save_dir, 'metrics.npy'), metrics)
    print("✅ 保存完成！")