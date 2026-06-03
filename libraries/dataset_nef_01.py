import os
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from PIL import Image

class CustomNEFPairDataset(Dataset):
    """
    Кастомный датасет для работы с 16-битными RAW (NEF) данными флэт-филдов.
    LQ на входе: 4-канальный тензор RGGB (матрица Байера).
    HQ на выходе: 3-канальный тензор RGB (чистый эталонный градиент).
    """
    def __init__(self, dataroot_lq, dataroot_gt):
        self.dataroot_lq = dataroot_lq
        self.dataroot_gt = dataroot_gt
        
        # Находим файлы по базовым именам (портилка сохраняет LQ как tiff, HQ как png или tiff)
        # Ищем совпадения по именам, отсекая расширения
        self.filenames = [os.path.splitext(f)[0].replace('_bayer', '') 
                          for f in os.listdir(dataroot_lq) if f.endswith('.tiff')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        
        # 1. Загружаем LQ (Зашумленная матрица Байера от портилки, 16-бит)
        lq_path = os.path.join(self.dataroot_lq, f"{name}_bayer.tiff")
        bayer = tifffile.imread(lq_path).astype(np.float32) / 65535.0
        
        # 🎯 ИСПРАВЛЕНИЕ: Гарантируем четность размеров перед нарезкой каналов!
        h, w = bayer.shape
        h = h - (h % 2)
        w = w - (w % 2)
        bayer = bayer[:h, :w] # Отрезаем нечетный крайний пиксель, если он был
        
        # Теперь нарезка гарантированно выдаст абсолютно одинаковые формы для всех 4 каналов
        r   = bayer[0::2, 0::2]
        g1  = bayer[0::2, 1::2]
        g2  = bayer[1::2, 0::2] # 🎯 Напоминание: тут мы зафиксировали порядок RGGB для Nikon!
        b   = bayer[1::2, 1::2]
        
        # Склеиваем в массив [H/2, W/2, 4]
        lq_rggb = np.stack([r, g1, g2, b], axis=2)
        # Переводим в тензор PyTorch [4, H/2, W/2]
        lq_tensor = torch.from_numpy(lq_rggb.transpose(2, 0, 1)).float()

        # 2. Загружаем HQ (Чистый дебайеризованный RGB эталон, 16-бит)
        gt_path = os.path.join(self.dataroot_gt, f"{name}.png")
        
        # Если портилка сохранила HQ как PNG, PIL прочитает его. 
        # Если как 16-битный TIFF, подстрахуемся проверкой:
        if not os.path.exists(gt_path):
            gt_path = os.path.join(self.dataroot_gt, f"{name}.tiff")
            gt_rgb = tifffile.imread(gt_path).astype(np.float32) / 65535.0
        else:
            # Читаем PNG (конвертируем в float32)
            gt_img = Image.open(gt_path).convert('RGB')
            gt_rgb = np.array(gt_img).astype(np.float32) / 255.0

        # Переводим HQ в тензор PyTorch [3, H, W]
        gt_tensor = torch.from_numpy(gt_rgb.transpose(2, 0, 1)).float()

        # 🎯 ВАЖНО: Модели NAFNet требуется, чтобы размеры LQ и HQ тензоров 
        # на этапе батчевания были фиксированными (например, 256x256).
        # Делаем быстрый ресайз средствами PyTorch functional:
        import torchvision.transforms.functional as F
        
        lq_tensor = F.resize(lq_tensor, (256, 256), interpolation=F.InterpolationMode.BILINEAR)
        gt_tensor = F.resize(gt_tensor, (256, 256), interpolation=F.InterpolationMode.BILINEAR)

        return lq_tensor, gt_tensor
