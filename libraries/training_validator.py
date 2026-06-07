import os
import sys
import cv2
import math
import torch
import numpy as np
import tifffile
from PIL import Image

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from libraries.logger import get_logger

class VisualValidator:
    def __init__(self, opt):
        self.opt = opt
        self.logger = get_logger()
        log_cfg = self.opt.get('logger', {}).get('include_in_file_log', {})
        self.log_ssim_to_file = log_cfg.get('log_ssim', False)
        self.log_console_cfg = self.opt.get('logger', {}).get('include_in_console', {})
        self.in_channels = opt['network_g']['num_in_ch']
        self.upscale_factor = opt.get('network_g', {}).get('upscale_factor', 2)
        exp_name = opt.get('name', 'default_experiment')
        
        dataset_root = opt['path']['dataset_root']
        self.hq_val_dir = os.path.join(dataset_root, "test", "hq_targets")
        self.lq_val_dir = os.path.join(dataset_root, "test", "lq_inputs")
        self.out_val_dir = os.path.join('experiments', exp_name, 'val_predictions')
        os.makedirs(self.out_val_dir, exist_ok=True)
        self.ext = '.jpg' if self.in_channels == 3 else '.tiff'
        
        # Динамически подтягиваем размеры патчей из YAML конфигурации
        train_cfg = opt.get('datasets', {}).get('train', {})
        self.gt_size = train_cfg.get('gt_size', 256)
        self.lq_size = train_cfg.get('lq_size', 128)

    def run_validation(self, model, epoch, device):
        """Прогоняет все тестовые файлы через модель с использованием честного Center Crop и расчётом PSNR/SSIM"""
        ssim_values = []
        psnr_values = []
        
        if not os.path.exists(self.lq_val_dir):
            self.logger.warning(f"Папка {self.lq_val_dir} не найдена. Пропускаем.")
            return 0.0, 0.0
            
        files = [f for f in os.listdir(self.lq_val_dir) if f.lower().endswith(self.ext)]
        if not files:
            self.logger.warning(f"Нет файлов с расширением {self.ext} в {self.lq_val_dir}")
            return 0.0, 0.0
            
        model.eval()
        
        with torch.no_grad():
            for file_name in files:
                file_path = os.path.join(self.lq_val_dir, file_name)
                base_name = os.path.splitext(file_name)[0].replace('_bayer', '')
                
                # Загружаем LQ RAW подканалы (H_raw x W_raw x 4)
                lq = tifffile.imread(file_path).astype(np.float32) / 65535.0
                h_lq, w_lq = lq.shape[:2]
                
                # Академический Center Crop на входе: вырезаем строго центральный квадрат lq_size
                top_lq = (h_lq - self.lq_size) // 2 if h_lq > self.lq_size else 0
                left_lq = (w_lq - self.lq_size) // 2 if w_lq > self.lq_size else 0
                lq_cropped = lq[top_lq:top_lq+self.lq_size, left_lq:left_lq+self.lq_size, :]
                
                lq_tensor = torch.from_numpy(lq_cropped.transpose(2, 0, 1)).float()
                input_tensor = lq_tensor.unsqueeze(0).to(device)

                # Инференс: сеть выдает честный RGB квадрат gt_size в полном разрешении
                output = model(input_tensor)
                if isinstance(output, dict):
                    output = output['out']

                output_np = output.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                final_img = (output_np * 255.0).astype(np.uint8)

                # Загружаем эталонное изображение HQ
                gt_path = os.path.join(self.hq_val_dir, f"{base_name}.png")
                if os.path.exists(gt_path):
                    gt_img = np.array(Image.open(gt_path).convert('RGB'))
                    
                    # Геометрическая привязка к нашему единому масштабу
                    top_hq = top_lq * self.upscale_factor
                    left_hq = left_lq * self.upscale_factor
                    
                    # 🎯 УБРАЛИ МУСОРНОЕ .copy()! Работаем через легковесный срез NumPy
                    gt_cropped = gt_img[top_hq:top_hq+self.gt_size, left_hq:left_hq+self.gt_size, :]
                    
                    # Замок геометрического совпадения ВАК
                    assert gt_cropped.shape[:2] == final_img.shape[:2], \
                        f"Рассинхронизация матриц! GT: {gt_cropped.shape[:2]}, Модель: {final_img.shape[:2]}"
                    
                    # Вычисляем честный SSIM
                    ssim_val = ssim(final_img, gt_cropped, channel_axis=2, data_range=255)
                    ssim_values.append(ssim_val)
                    
                    # Вычисляем честный PSNR в пространстве uint8 [0, 255]
                    mse_val = np.mean((final_img.astype(np.float32) - gt_cropped.astype(np.float32)) ** 2)
                    if mse_val == 0:
                        psnr_val = 100.0
                    else:
                        psnr_val = 20 * math.log10(255.0 / math.sqrt(mse_val))
                    psnr_values.append(psnr_val)
                    
                    if self.log_ssim_to_file:
                        self.logger.info(f"[Валидатор] {base_name}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}")
                
                # Сохраняем верификационную картинку на диск
                out_name = f"epoch_{epoch}_{base_name}.png"
                out_path = os.path.join(self.out_val_dir, out_name)
                Image.fromarray(final_img).save(out_path)
        
        model.train()
        mean_ssim = np.mean(ssim_values) if ssim_values else 0.0
        mean_psnr = np.mean(psnr_values) if psnr_values else 0.0
        
        self.logger.info(f"📊 [Валидация] Итог эпохи {epoch} — Средний PSNR: {mean_psnr:.2f} dB, Средний SSIM: {mean_ssim:.4f}")
        
        return mean_ssim, mean_psnr


    def calculate_ssim(self, restored_path, gt_path):
        restored = np.array(Image.open(restored_path).convert('RGB'))
        gt = np.array(Image.open(gt_path).convert('RGB'))
        return ssim(restored, gt, channel_axis=2, data_range=255)
