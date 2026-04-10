import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
from tqdm import tqdm
import pickle
import logging
import numpy as np

# ==============================================================================
# 🚑 Numpy 2.x 兼容性补丁 (必须放在最前面)
# ==============================================================================
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

# ==============================================================================
# 📦 导入自定义模块
# ==============================================================================
# 确保你的 model.py 在同一目录下
from openstl.models.earthfarseer_model import Earthfarseer_model
# 确保 utils_exp.py 和 test_engine_standard.py 在同一目录下
from openstl.utils.ef_utils_exp import create_exp_dir, setup_logger
from openstl.utils.ef_test_engine import AgriSTLTester

# 全局 Logger (初始化后会被赋值)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. 数据加载器 (经过极致优化)
# ==============================================================================
def load_plant_pkl_data(path, batch_size=4):
    """
    加载 AgriSTL 风格的 .pkl 数据文件
    """
    logger.info(f"正在加载数据: {path} ...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # 提取数据 (假设数据已经是 float32 [0,1])
    train_x = torch.from_numpy(data['X_train'])
    train_y = torch.from_numpy(data['Y_train'])
    val_x = torch.from_numpy(data['X_val'])
    val_y = torch.from_numpy(data['Y_val'])
    test_x = torch.from_numpy(data['X_test'])
    test_y = torch.from_numpy(data['Y_test'])

    logger.info(f"✅ 数据加载成功!")
    logger.info(f"   Train Shape: X {train_x.shape}, Y {train_y.shape}")
    logger.info(f"   Val   Shape: X {val_x.shape}, Y {val_y.shape}")

    # 🚀 DataLoader 性能优化配置
    loader_args = dict(
        batch_size=batch_size,
        num_workers=8,  # 建议设置为 CPU 核心数 (如 4, 8)
        pin_memory=True,  # 开启锁页内存，加速 CPU->GPU
        persistent_workers=True,  # 避免每轮重建进程
        prefetch_factor=2
    )

    train_loader = DataLoader(TensorDataset(train_x, train_y), shuffle=True, **loader_args)
    # 验证和测试不需要 shuffle
    eval_loader = DataLoader(TensorDataset(val_x, val_y), shuffle=False, **loader_args)
    test_loader = DataLoader(TensorDataset(test_x, test_y), shuffle=False, **loader_args)

    return train_loader, eval_loader, test_loader


# ==============================================================================
# 2. 验证函数 (Validation)
# ==============================================================================
def evaluate_model(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            # 这里的 non_blocking 配合 pin_memory 可以保留，加速数据传输
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # ❌ 删掉 with autocast():
            # ✅ 直接运行 (float32)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples


# ==============================================================================
# 3. 训练主循环 (集成 AMP 和 Logging)
# ==============================================================================
def train_model(model, train_loader, eval_loader, criterion, optimizer, device, num_epochs, ckpt_dir):
    best_loss = float('inf')
    best_model_path = os.path.join(ckpt_dir, 'best.pth')

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # (已移除 Scaler 初始化)

    logger.info(f"🚀 Start Training Loop for {num_epochs} epochs (FP32 Mode)...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            # 1. 前向传播 (FP32)
            preds = model(inputs)
            loss = criterion(preds, targets)

            # 2. 反向传播
            loss.backward()

            # 3. 梯度裁剪 (关键修改：建议设为 1.0 以防止 NaN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 4. 更新参数
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())

        # 计算平均 Loss
        average_loss = total_loss / total_samples

        # 验证
        eval_loss = evaluate_model(model, eval_loader, criterion, device)
        scheduler.step(eval_loss)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 记录日志
        logger.info(
            f'Epoch {epoch + 1} | Train Loss: {average_loss:.6f} | Val Loss: {eval_loss:.6f} | LR: {current_lr:.6f}')

        # 保存最佳模型
        if eval_loss < best_loss:
            best_loss = eval_loss
            logger.info(f'⭐️ New best found ({best_loss:.6f})! Saving to {best_model_path}...')
            torch.save(model.state_dict(), best_model_path)

    logger.info("Training complete.")
    return best_model_path


# ==============================================================================
# 4. Main 入口
# ==============================================================================
def main(args):
    # ----------------------------------------------------------
    # 1. 📂 环境与目录设置 (AgriSTL Style)
    # ----------------------------------------------------------
    global logger
    exp_dirs = create_exp_dir(args)  # 创建 outputs/Exp_Time/
    logger = setup_logger(exp_dirs['logs'])  # 设置日志到文件和控制台

    logger.info(f"Args Configuration: {args}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    # ----------------------------------------------------------
    # 2. 💿 加载数据
    # ----------------------------------------------------------
    train_loader, eval_loader, test_loader = load_plant_pkl_data(
        path=args.data_path,
        batch_size=args.batch_size
    )

    # 自动获取数据维度 (B, T, C, H, W)
    sample_input, _ = next(iter(train_loader))
    _, _, C, H, W = sample_input.shape
    logger.info(f"Detected Data Shape: Channel={C}, H={H}, W={W}")

    # ----------------------------------------------------------
    # 3. 🏗️ 初始化模型
    # ----------------------------------------------------------
    # 注意：args.T_in 必须与数据一致
    model = Earthfarseer_model(shape_in=(args.T_in, C, H, W))
    model.to(device)

    # 🔥 PyTorch 2.x 编译加速 (如果报错会自动回退)
    # try:
    #     logger.info("🚀 Compiling model with torch.compile ...")
    #     model = torch.compile(model)
    # except Exception as e:
    #     logger.warning(f"⚠️ torch.compile failed: {e}. Running in eager mode.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------------------------------------------
    # 4. 🏋️ 开始训练
    # ----------------------------------------------------------
    logger.info(">>> Start Training Phase <<<")
    best_ckpt_path = train_model(
        model, train_loader, eval_loader, criterion, optimizer, device,
        num_epochs=args.num_epochs,
        ckpt_dir=exp_dirs['ckpt']
    )

    # ----------------------------------------------------------
    # 5. 🧪 最终测试 (AgriSTL 接口对齐)
    # ----------------------------------------------------------
    logger.info("\n>>> Start Testing Phase (AgriSTL Standard) <<<")

    # 加载最佳权重
    logger.info(f"Loading best weights from: {best_ckpt_path}")
    state_dict = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # 实例化标准化测试器
    # 结果 (npy) 将被保存到 exp_dirs['saved'] -> outputs/.../saved/
    tester = AgriSTLTester(model, device, save_dir=exp_dirs['saved'])

    # 运行测试 (自动保存 inputs.npy, preds.npy, trues.npy, metrics.npy)
    metrics = tester.test(test_loader)

    logger.info(f"🎉 All Done! Results saved in {exp_dirs['root']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Earthfarseer for Plant Growth (AgriSTL Compatible)")

    # 数据路径
    parser.add_argument('--data_path', type=str,
                        default='/data/user4ddl/data/PlantGrwothPredction_data/TomatoData/ProcessedData/pre5_aft5_192x128_data/pkl_data/pre5_aft5_192x128_aug.pkl',
                        help='Path to the .pkl dataset.')

    # 实验名称 (用于生成文件夹 outputs/Tomato_Exp_xxxx)
    parser.add_argument('--ex_name', type=str, default='Tomato_Exp', help='Experiment name prefix')

    # 模型参数
    parser.add_argument('--T_in', type=int, default=5, help='Input sequence length')
    parser.add_argument('--T_out', type=int, default=5, help='Output sequence length')

    # 训练参数
    # 建议开启混合精度后 batch_size 可以设为 8 或 16
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')

    args = parser.parse_args()
    main(args)