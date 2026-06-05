import os
import csv
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from libraries.logger import get_logger

logger = get_logger()

class TrainingLogger:
    def __init__(self, opt, save_dir):
        self.opt = opt
        self.save_dir = save_dir
        self.use_tb = opt.get('logger', {}).get('use_tb_logger', False)
        self.tb_logger = None
        self.prev_psnr = None   # для delta PSNR

        # Инициализация TensorBoard
        if self.use_tb:
            log_dir = os.path.join(save_dir, 'tb_logs')
            self.tb_logger = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard инициализирован в: {log_dir}")
        else:
            logger.info("TensorBoard отключен в настройках YAML.")

        # ---- CSV логирование (расширенное) ----
        self.log_csv = opt.get('logger', {}).get('log_csv', False)
        self.log_loss_components = opt.get('logger', {}).get('log_loss_components', True)

        if self.log_csv:
            csv_filename = opt.get('logger', {}).get('csv_file_name', 'train_metrics.csv')
            self.csv_path = os.path.join(save_dir, csv_filename)
            self.csv_headers = [
                'step', 'loss', 'psnr', 'lr',
                'delta_psnr', 'grad_var', 'tv_ratio',
                'vram_alloc_gb', 'vram_res_gb',
                'l1_loss', 'ffl_loss'
            ]
            # Создаём файл с заголовками, если его нет
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                logger.info(f"CSV лог создан: {self.csv_path}")

    def log_metrics(self, loss_val, psnr_val, lr_val, global_step,
                    model=None, targets=None, outputs=None, device=None,
                    l1_loss_val=None, ffl_loss_val=None):
        """
        Логирует метрики в TensorBoard, текстовый файл и CSV.
        l1_loss_val, ffl_loss_val - дополнительные компоненты комбинированной потери.
        """
        # 1. TensorBoard
        if self.use_tb and self.tb_logger is not None:
            self.tb_logger.add_scalar('train/loss', loss_val, global_step)
            self.tb_logger.add_scalar('train/psnr', psnr_val, global_step)
            self.tb_logger.add_scalar('train/lr', lr_val, global_step)

        # 2. Подготовка данных для логов
        logger_opt = self.opt.get('logger', {})
        file_freq = logger_opt.get('file_log_freq', 1)
        delta_psnr = 0.0
        grad_var = 0.0
        tv_ratio = 0.0
        alloc_vram = 0.0
        res_vram = 0.0

        # delta PSNR
        if self.prev_psnr is not None:
            delta_psnr = psnr_val - self.prev_psnr
        self.prev_psnr = psnr_val

        # Grad variance
        if model is not None:
            grad_norms = [p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None]
            grad_var = np.var(grad_norms) if grad_norms else 0.0

        # TV ratio
        if targets is not None and outputs is not None:
            tv_out = torch.sum(torch.abs(outputs[:, :, :, :-1] - outputs[:, :, :, 1:])) + \
                     torch.sum(torch.abs(outputs[:, :, :-1, :] - outputs[:, :, 1:, :]))
            tv_tar = torch.sum(torch.abs(targets[:, :, :, :-1] - targets[:, :, :, 1:])) + \
                     torch.sum(torch.abs(targets[:, :, :-1, :] - targets[:, :, 1:, :]))
            tv_ratio = (tv_out / (tv_tar + 1e-8)).item()

        # VRAM usage
        if device is not None and torch.cuda.is_available():
            alloc_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
            res_vram = torch.cuda.memory_reserved(device) / (1024 ** 3)

        # 3. Запись в текстовый файл (train_progress.log)
        if global_step % file_freq == 0:
            log_file_name = logger_opt.get('log_file_name', 'train_progress.log')
            log_path = os.path.join(self.save_dir, log_file_name)
            flags = logger_opt.get('include_in_file_log', {})

            log_parts = [f"[Шаг: {global_step:05d}]"]
            if flags.get('loss', True):
                log_parts.append(f"Loss: {loss_val:.6f}")
            if flags.get('psnr', True):
                log_parts.append(f"PSNR: {psnr_val:.2f} dB")
            if flags.get('learning_rate', True):
                log_parts.append(f"LR: {lr_val:.2e}")
            if flags.get('delta_psnr', True):
                log_parts.append(f"dPSNR: {delta_psnr:+.4f}")
            if flags.get('grad_variance', True):
                log_parts.append(f"GradVar: {grad_var:.2e}")
            if flags.get('tv_ratio', True):
                log_parts.append(f"TV_Ratio: {tv_ratio:.3f}")
            if flags.get('vram_usage', True):
                log_parts.append(f"VRAM: {alloc_vram:.2f}GB/{res_vram:.2f}GB")
            # Дополнительные компоненты лосса (если есть и включены)
            if self.log_loss_components and flags.get('l1_loss', False) and l1_loss_val is not None:
                log_parts.append(f"L1: {l1_loss_val:.6f}")
            if self.log_loss_components and flags.get('ffl_loss', False) and ffl_loss_val is not None:
                log_parts.append(f"FFL: {ffl_loss_val:.6f}")

            log_string = " | ".join(log_parts) + "\n"
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_string)

        # 4. Запись в CSV (всегда полный набор колонок)
        if self.log_csv:
            l1 = l1_loss_val if l1_loss_val is not None else 0.0
            ffl = ffl_loss_val if ffl_loss_val is not None else 0.0
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step, loss_val, psnr_val, lr_val,
                    delta_psnr, grad_var, tv_ratio, alloc_vram, res_vram,
                    l1, ffl
                ])

    def log_validation_image(self, tag, image_tensor, epoch):
        if self.use_tb and self.tb_logger:
            self.tb_logger.add_image(tag, image_tensor, epoch)

    def close(self):
        if self.tb_logger:
            self.tb_logger.close()
            logger.info("Потоки TensorBoard закрыты.")

    def log_validation_metrics(self, epoch, ssim, psnr_val=None):
        val_csv_path = os.path.join(self.save_dir, 'val_metrics.csv')
        file_exists = os.path.exists(val_csv_path)
        with open(val_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                if psnr_val is not None:
                    writer.writerow(['epoch', 'ssim', 'psnr'])
                else:
                    writer.writerow(['epoch', 'ssim'])
            if psnr_val is not None:
                writer.writerow([epoch, ssim, psnr_val])
            else:
                writer.writerow([epoch, ssim])
        logger.info(f"Валидационные метрики сохранены в {val_csv_path}")