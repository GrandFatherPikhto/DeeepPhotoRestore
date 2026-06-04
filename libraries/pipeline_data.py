import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image
import albumentations as A
from tqdm import tqdm
from libraries.pipeline_logger import get_logger

logger = get_logger()

class RestorationDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.is_train = is_train
        dataset_root = config.get("dataset_root", "datasets/nef_nafnet")
        phase = "train" if is_train else "val"
        self.lq_dir = os.path.join(dataset_root, phase, "lq_inputs")
        self.hq_dir = os.path.join(dataset_root, phase, "hq_targets")
        data_cfg = config.get("datasets", {}).get(phase, {})
        self.gt_size = data_cfg.get("gt_size", 256)
        
        self.file_names = sorted(os.listdir(self.lq_dir))
        logger.info(f"Загрузка датасета, {len(self.file_names)} файлов")
        
        # Геометрические аугментации (трансформации, не меняющие число каналов)
        # Для LQ (4 канала) и HQ (3 канала) нужно применять одинаковые трансформации
        # Пока отключим для простоты, включим позже
        self.use_augment = False  # временно отключаем аугментации
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
                # Читаем LQ как 4-канальный TIFF (H, W, 4)
                lq_img = tifffile.imread(lq_path).astype(np.float32) / 65535.0
                # Читаем HQ как RGB PNG (H, W, 3)
                hq_img = np.array(Image.open(hq_path).convert('RGB')).astype(np.float32) / 255.0
                if lq_img is not None and hq_img is not None:
                    break
            except Exception as e:
                logger.warning(f"Ошибка чтения {lq_path}: {e}")
                idx = random.randint(0, len(self.file_names) - 1)
        else:
            raise FileNotFoundError("All files unreadable")

        # Приведение к размеру gt_size
        lq_img, hq_img = self._resize_to_gt(lq_img, hq_img)

        # Аугментации (отключены)
        if self.geom_transform is not None:
            # Для LQ и HQ нужно применять одни и те же параметры трансформации
            # Пока пропускаем, так как число каналов разное (4 vs 3)
            # В будущем можно использовать параметризованные трансформации
            pass

        # Преобразование в тензоры
        lq_tensor = torch.from_numpy(lq_img.transpose(2, 0, 1)).float()   # (4, H, W)
        hq_tensor = torch.from_numpy(hq_img.transpose(2, 0, 1)).float()   # (3, H, W)
        return lq_tensor, hq_tensor

    def _resize_to_gt(self, lq_img, hq_img):
        gt_size = self.gt_size
        h, w = lq_img.shape[:2]
        if h == gt_size and w == gt_size:
            return lq_img, hq_img
        from skimage.transform import resize
        lq_resized = resize(lq_img, (gt_size, gt_size), preserve_range=True, anti_aliasing=True).astype(lq_img.dtype)
        hq_resized = resize(hq_img, (gt_size, gt_size), preserve_range=True, anti_aliasing=True).astype(hq_img.dtype)
        return lq_resized, hq_resized

def create_restoration_dataset(config, is_train=True):
    return RestorationDataset(config, is_train=is_train)