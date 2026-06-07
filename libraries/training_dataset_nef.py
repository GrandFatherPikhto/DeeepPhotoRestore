import os
import sys
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from PIL import Image

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torchvision.transforms.functional as TF


class CustomNEFPairDataset(Dataset):
    def __init__(self, dataroot_lq, dataroot_gt, opt=None):
        self.dataroot_lq = dataroot_lq
        self.dataroot_gt = dataroot_gt
        
        # Формируем список базовых имён (без расширений)
        self.filenames = [
            os.path.splitext(f)[0].replace('_bayer', '')
            for f in os.listdir(dataroot_lq)
            if f.endswith('.tiff')
        ]
        
        # Задаём единственный источник истины для масштабирования (Модель -> Датасет)
        if opt is None:
            opt = {}
        self.upscale_factor = opt.get('network_g', {}).get('upscale_factor', 2)
        
        train_cfg = opt.get('datasets', {}).get('train', {})
        self.gt_size = train_cfg.get('gt_size', 256)   # размер HQ (должен быть upscale_factor * lq_size)
        self.lq_size = train_cfg.get('lq_size', 128)
        self.use_flip = train_cfg.get('use_flip', False)
        self.use_rot = train_cfg.get('use_rot', False)

        # Жёсткая верификация геометрической целостности на базе единого источника истины
        assert self.gt_size == self.upscale_factor * self.lq_size, \
            f"Критическая ошибка геометрии ВАК: gt_size ({self.gt_size}) должен быть строго равен " \
            f"upscale_factor ({self.upscale_factor}) * lq_size ({self.lq_size})!"        
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        
        # Загружаем LQ (4 канала, uint16, упакованный RGGB)
        lq_path = os.path.join(self.dataroot_lq, f"{name}_bayer.tiff")
        lq_packed = tifffile.imread(lq_path).astype(np.float32) / 65535.0   # (H_lq, W_lq, 4)
        
        # Загружаем HQ (RGB, uint8, полноразмерный резкий таргет)
        gt_path = os.path.join(self.dataroot_gt, f"{name}.png")
        gt_img = Image.open(gt_path).convert('RGB')
        gt_rgb = np.array(gt_img).astype(np.float32) / 255.0                # (H_hq, W_hq, 3)
        
        h_lq, w_lq = lq_packed.shape[:2]
        h_hq, w_hq = gt_rgb.shape[:2]
        
        # Академическая проверка: размеры HQ должны строго соответствовать масштабу апскейла
        assert h_hq == self.upscale_factor * h_lq and w_hq == self.upscale_factor * w_lq, \
            f"Рассинхронизация размеров файлов для {name}: LQ={lq_packed.shape}, HQ={gt_rgb.shape}"
        
        # --- Академический случайный кроп (одинаковая геометрия для LQ и HQ) ---
        top_lq = random.randint(0, h_lq - self.lq_size) if h_lq > self.lq_size else 0
        left_lq = random.randint(0, w_lq - self.lq_size) if w_lq > self.lq_size else 0
        lq_cropped = lq_packed[top_lq:top_lq+self.lq_size, left_lq:left_lq+self.lq_size, :]
        
        # Честная геометрическая привязка к масштабу апскейла
        top_hq = top_lq * self.upscale_factor
        left_hq = left_lq * self.upscale_factor
        
        # Вырезаем эталонный патч строго из соответствующего места сцены
        gt_cropped = gt_rgb[top_hq:top_hq+self.gt_size, left_hq:left_hq+self.gt_size, :]
        
        # Конвертируем в тензоры PyTorch (HWC -> CHW)
        lq_tensor = torch.from_numpy(lq_cropped.transpose(2, 0, 1)).float()   # (4, lq_size, lq_size)
        gt_tensor = torch.from_numpy(gt_cropped.transpose(2, 0, 1)).float()   # (3, gt_size, gt_size)
        
        # Безопасные для Bayer-матрицы аугментации (только отражения!)
        if self.use_flip:
            if random.random() > 0.5:
                lq_tensor = TF.hflip(lq_tensor)
                gt_tensor = TF.hflip(gt_tensor)
            if random.random() > 0.5:
                lq_tensor = TF.vflip(lq_tensor)
                gt_tensor = TF.vflip(gt_tensor)
                
        return lq_tensor, gt_tensor
