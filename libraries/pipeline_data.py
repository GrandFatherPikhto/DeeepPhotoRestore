import os
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image
import albumentations as A
from skimage.transform import resize
from libraries.logger import get_logger

logger = get_logger()

class RestorationDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.is_train = is_train
        path = config.get('path', {})
        dataset_root = path.get("dataset_root", "datasets/nef_nafnet")
        phase = "train" if is_train else "val"
        self.lq_dir = os.path.join(dataset_root, phase, "lq_inputs")
        self.hq_dir = os.path.join(dataset_root, phase, "hq_targets")
        data_cfg = config.get("datasets", {}).get(phase, {})
        self.gt_size = data_cfg.get("gt_size", 256)
        self.lq_size = data_cfg.get("lq_size", 128)
        
        self.file_names = sorted(os.listdir(self.lq_dir))
        logger.info(f"Загрузка датасета, {len(self.file_names)} файлов")
        
        self.use_augment = True
        if self.use_augment and self.is_train:
            self.geom_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            self.geom_transform = None

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        import random
        for _ in range(len(self.file_names)):
            lq_name = self.file_names[idx]
            hq_name = lq_name.replace("_bayer.tiff", ".png")
            lq_path = os.path.join(self.lq_dir, lq_name)
            hq_path = os.path.join(self.hq_dir, hq_name)
            try:
                lq_img = tifffile.imread(lq_path).astype(np.float32) / 65535.0
                hq_img = np.array(Image.open(hq_path).convert('RGB')).astype(np.float32) / 255.0
                if lq_img is None or hq_img is None:
                    raise ValueError
                break
            except Exception as e:
                logger.warning(f"Ошибка чтения {lq_path}: {e}")
                idx = random.randint(0, len(self.file_names) - 1)
        else:
            raise FileNotFoundError("All files unreadable")

        # Принудительный ресайз до целевых размеров
        lq_img = resize(lq_img, (self.lq_size, self.lq_size), preserve_range=True, anti_aliasing=True).astype(lq_img.dtype)
        hq_img = resize(hq_img, (self.gt_size, self.gt_size), preserve_range=True, anti_aliasing=True).astype(hq_img.dtype)

        # [Никаких циклов изменения размера здесь больше нет!]
        # Данные загружены в try-except блоке, они уже имеют правильный физический масштаб.
        
        if self.geom_transform is not None:
            # Преобразуем в формат, понятный Albumentations (если это необходимо)
            # Но помни: повороты на 90 градусов мы отключили в конфиге!
            pass
            
        # Честное копирование без интерполяционных искажений
        lq_tensor = torch.from_numpy(lq_img.transpose(2, 0, 1)).float()  # (4, H_lq, W_lq)
        hq_tensor = torch.from_numpy(hq_img.transpose(2, 0, 1)).float()  # (3, H_hq, W_hq)
        
        return lq_tensor, hq_tensor


def create_restoration_dataset(config, is_train=True):
    return RestorationDataset(config, is_train=is_train)