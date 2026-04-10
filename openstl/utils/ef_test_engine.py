import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from openstl.utils.ef_utils_exp import save_agristl_results
import logging


class AgriSTLTester:
    def __init__(self, model, device, save_dir):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.logger = logging.getLogger(__name__)

    def test(self, test_loader):
        self.model.eval()

        inputs_list = []
        trues_list = []
        preds_list = []

        self.logger.info(f"🚀 开始测试 (Dataset size: {len(test_loader.dataset)})")

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Testing Inference"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 1. 推理
                preds = self.model(inputs)
                preds = torch.clamp(preds, 0.0, 1.0)

                # 2. 收集数据 (保持在内存中，最后合并)
                inputs_list.append(inputs.cpu().numpy())
                trues_list.append(targets.cpu().numpy())
                preds_list.append(preds.cpu().numpy())

        # 3. 拼接所有 Batch
        inputs_np = np.concatenate(inputs_list, axis=0)
        trues_np = np.concatenate(trues_list, axis=0)
        preds_np = np.concatenate(preds_list, axis=0)

        self.logger.info(f"数据收集完毕. Inputs shape: {inputs_np.shape}")

        # 4. 计算指标
        metrics = self._compute_metrics(trues_np, preds_np)

        self._log_metrics(metrics)

        # 5. 保存结果
        save_agristl_results(self.save_dir, inputs_np, trues_np, preds_np, metrics)

        return metrics

    def _compute_metrics(self, trues, preds):
        """
        计算指标，并对齐 AgriSTL 的计算方式 (Scale back to 0-255)
        trues, preds: (N, T, C, H, W) Numpy Arrays, Range [0, 1]
        """
        self.logger.info("正在计算指标 (已对齐 AgriSTL 0-255 尺度)...")

        # 1. 核心修改：反归一化到 0-255
        # AgriSTL 的 MSE 是基于像素值的
        preds_255 = preds * 255.0
        trues_255 = trues * 255.0

        # --- MSE & MAE (基于 0-255) ---
        mse = np.mean((preds_255 - trues_255) ** 2)
        mae = np.mean(np.abs(preds_255 - trues_255))

        # --- PSNR ---
        # 峰值是 255.0
        rmse = np.sqrt(mse)
        if rmse == 0:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(255.0 / rmse)

        # --- SSIM ---
        # SSIM 函数如果不指定 data_range，它会根据输入图像推断
        # 但为了保险，既然我们输入的是 0-255 的 float，最好指定 data_range=255
        ssim_val = 0.0
        N, T, C, H, W = preds.shape
        total_frames = N * T

        # 展平以便遍历
        flat_preds = preds_255.reshape(-1, C, H, W)
        flat_trues = trues_255.reshape(-1, C, H, W)

        # 采样计算 (为了快一点，你可以只算前 1000 帧，或者全算)
        # 这里演示全算
        for i in range(total_frames):
            p = flat_preds[i].transpose(1, 2, 0)  # H,W,C
            t_gt = flat_trues[i].transpose(1, 2, 0)

            # 关键：data_range=255.0
            ssim_val += ssim(t_gt, p, data_range=255.0, channel_axis=-1, win_size=7)

        avg_ssim = ssim_val / total_frames

        return {'mse': mse, 'mae': mae, 'ssim': avg_ssim, 'psnr': psnr}

    def _log_metrics(self, metrics):
        self.logger.info("\n" + "=" * 40)
        self.logger.info("📊 Final Test Results")
        self.logger.info("=" * 40)
        self.logger.info(f"📉 MSE  : {metrics['mse']:.6f}")
        self.logger.info(f"📉 MAE  : {metrics['mae']:.6f}")
        self.logger.info(f"📸 PSNR : {metrics['psnr']:.4f} dB")  # 新增
        self.logger.info(f"📈 SSIM : {metrics['ssim']:.6f}")
        self.logger.info("=" * 40)