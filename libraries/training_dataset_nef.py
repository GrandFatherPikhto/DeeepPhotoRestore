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
        
        # Читаем параметры из конфига
        train_cfg = opt.get('datasets', {}).get('train', {}) if opt else {}
        # self.upscale_factor = opt.get('datasets', {}).get('train', {}).get('upscale_factor', 2)
        self.upscale_factor = opt.get('datasets', {}).get('train', {}).get('upscale_factor', 2)
        self.gt_size = train_cfg.get('gt_size', 256)   # размер HQ (должен быть 2 * lq_size)
        self.lq_size = train_cfg.get('lq_size', 128)
        self.use_flip = train_cfg.get('use_flip', False)
        self.use_rot = train_cfg.get('use_rot', False)
        
        # Проверка соотношения размеров
        assert self.gt_size == self.upscale_factor * self.lq_size, \
            f"gt_size ({self.gt_size}) must be {self.upscale_factor} * lq_size ({self.lq_size})"

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
        
        # Проверка соотношения масштабов (HQ должно быть в 2 раза больше LQ)
        # if h_hq != 2 * h_lq or w_hq != 2 * w_lq:
        #     # Если нет – делаем ресайз HQ к правильному размеру (аварийно)
        #     from skimage.transform import resize
        #     new_h, new_w = 2 * h_lq, 2 * w_lq
        #     gt_rgb = resize(gt_rgb, (new_h, new_w), preserve_range=True)
        #     h_hq, w_hq = new_h, new_w
        target_h = self.upscale_factor * h_lq
        target_w = self.upscale_factor * w_lq
        if h_hq != target_h or w_hq != target_w:
            from skimage.transform import resize
            gt_rgb = resize(gt_rgb, (target_h, target_w), preserve_range=True)
            h_hq, w_hq = target_h, target_w        
        
        # --- Академический случайный кроп (одинаковая геометрия для LQ и HQ) ---
        # Кроп LQ размера lq_size x lq_size
        top_lq = random.randint(0, h_lq - self.lq_size) if h_lq > self.lq_size else 0
        left_lq = random.randint(0, w_lq - self.lq_size) if w_lq > self.lq_size else 0
        lq_cropped = lq_packed[top_lq:top_lq+self.lq_size, left_lq:left_lq+self.lq_size, :]
        
        # Соответствующий кроп HQ (масштаб 2:1)
        top_hq = top_lq * 2
        left_hq = left_lq * 2
        gt_cropped = gt_rgb[top_hq:top_hq+self.gt_size, left_hq:left_hq+self.gt_size, :]
        
        # --- Применение аугментаций (синхронно для LQ и HQ) ---
        # Преобразуем в тензоры PyTorch (C, H, W)
        lq_tensor = torch.from_numpy(lq_cropped.transpose(2, 0, 1)).float()   # (4, lq_size, lq_size)
        gt_tensor = torch.from_numpy(gt_cropped.transpose(2, 0, 1)).float()   # (3, gt_size, gt_size)
        
        # 1. Случайное отражение по горизонтали и вертикали
        if self.use_flip:
            if random.random() > 0.5:
                lq_tensor = TF.hflip(lq_tensor)
                gt_tensor = TF.hflip(gt_tensor)
            if random.random() > 0.5:
                lq_tensor = TF.vflip(lq_tensor)
                gt_tensor = TF.vflip(gt_tensor)
        
        # 2. Случайный поворот на 90, 180 или 270 градусов
        if self.use_rot:
            k = random.randint(0, 3)   # 0 – без поворота, 1 – 90°, 2 – 180°, 3 – 270°
            if k > 0:
                lq_tensor = TF.rotate(lq_tensor, k * 90)
                gt_tensor = TF.rotate(gt_tensor, k * 90)
        
        return lq_tensor, gt_tensor