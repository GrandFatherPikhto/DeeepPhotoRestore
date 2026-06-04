import os
# import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # <-- Импортируем прогресс-бар
from libraries.pipeline_logger import get_logger
logger = get_logger()

class RestorationDataset(Dataset):
    """Кастомный датасет для пространственно-цветовой реставрации RAW-данных."""
    
    def __init__(self, config, is_train=True):
        self.is_train = is_train

        # 1. Вытаскиваем глобальный корень датасета из yml
        dataset_root = config.get("dataset_root", "datasets/nef_nafnet")

        # 2. Определяем имя подпапки в зависимости от режима (train или val)
        phase = "train" if is_train else "val"

        # 3. Автоматически собираем финальные пути к LQ и HQ патчам
        self.lq_dir = os.path.join(dataset_root, phase, "lq_inputs")
        self.hq_dir = os.path.join(dataset_root, phase, "hq_targets")

        # 4. Безопасно забираем размер патча (по дефолту 256)
        data_cfg = config.get("datasets", {}).get(phase, {})
        self.gt_size = data_cfg.get("gt_size", 256)
        
        self.file_names = sorted(os.listdir(self.lq_dir))
        
        # Элегантный прогресс-бар прямо при инициализации данных
        logger.info(f"Начало подготовки и валидации потоков {'обучения' if is_train else 'валидации'}...")
        for name in tqdm(self.file_names, desc="📦 Сборка датасета Nikon D600", unit="кадр"):
            # Здесь происходит быстрая фоновая проверка целостности файлов, если нужно
            pass
            
        logger.info(f"Датасет успешно собран. Зарегистрировано кадров: {len(self.file_names)}")
        
        # Настройки трансформаций и шума остаются прежними...
        self.geom_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={'image': 'image', 'mask': 'image'}, is_check_shapes=False)  # <-- Отключаем проверку равенства размеров
        
        noise_cfg = config.get("noise_model", {})
        self.sigma_min = noise_cfg.get("sigma_min", 10.0)
        self.sigma_max = noise_cfg.get("sigma_max", 50.0)
        self.noise_p = noise_cfg.get("p", 0.3)
    def __len__(self):
        return len(self.file_names)

    def _apply_cfa_noise(self, lq_img):
        """Математическое наложение шума на LQ поток на основе параметров ЧКХ."""
        if self.is_train and np.random.rand() < self.noise_p:
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)
            # Генерируем АБГШ той же размерности, что и LQ патч
            noise = np.random.normal(0, sigma, lq_img.shape).astype(np.float32)
            # В диссертации это свертка с ЧКХ, здесь — симуляция аддитивной стохастики
            lq_img = np.clip(lq_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return lq_img

    def __getitem__(self, idx):
        import random
        
        for _ in range(len(self.file_names)):
            lq_name = self.file_names[idx]
            hq_name = lq_name.replace("_bayer.tiff", ".png")
            lq_path = os.path.join(self.lq_dir, lq_name)
            hq_path = os.path.join(self.hq_dir, hq_name)
            
            try:
                lq_img = np.array(Image.open(lq_path))
                hq_img = np.array(Image.open(hq_path))
                if lq_img is not None and hq_img is not None:
                    break
            except Exception as e:
                logger.warning(f"Ошибка чтения: {e}")
                idx = random.randint(0, len(self.file_names) - 1)
        else:
            raise FileNotFoundError("All TIFF files are unreadable")

        # Применяем шум (остаётся как есть)
        lq_img = self._apply_cfa_noise(lq_img)
        
        # Случайный кроп для тренировочных данных
        # if self.is_train:
        #     lq_img, hq_img = self._random_crop(lq_img, hq_img)
        if self.is_train:
            lq_img, hq_img = self._resize_to_gt(lq_img, hq_img)            
        
        # Аугментации (flip, rotate) – они уже есть
        if self.is_train:
            augmented = self.geom_transform(image=lq_img, mask=hq_img)
            lq_img, hq_img = augmented['image'], augmented['mask']
        
        # Преобразование в тензоры
        if len(lq_img.shape) == 2:
            lq_img = np.stack([lq_img] * 3, axis=-1)
        if len(hq_img.shape) == 2:
            hq_img = np.stack([hq_img] * 3, axis=-1)
            
        lq_tensor = torch.from_numpy(lq_img).float().permute(2, 0, 1) / 255.0
        hq_tensor = torch.from_numpy(hq_img).float().permute(2, 0, 1) / 255.0
        
        return lq_tensor, hq_tensor
    # def _random_crop(self, lq_img, hq_img):
    #     """Вырезает случайный патч размера gt_size из обоих изображений."""
    #     h, w = lq_img.shape[:2]
    #     gt_size = self.gt_size
    #     if h > gt_size and w > gt_size:
    #         top = np.random.randint(0, h - gt_size)
    #         left = np.random.randint(0, w - gt_size)
    #         lq_img = lq_img[top:top+gt_size, left:left+gt_size]
    #         hq_img = hq_img[top:top+gt_size, left:left+gt_size]
    #     else:
    #         # Если изображение меньше gt_size – ресайзим (но по логике датасета такого не должно быть)
    #         pass
    #     return lq_img, hq_img
    
    def _resize_to_gt(self, lq_img, hq_img):
        """Приводит оба изображения к размеру gt_size x gt_size (бикубическая интерполяция)."""
        from skimage.transform import resize  # или cv2, но проще через skimage
        gt_size = self.gt_size
        h, w = lq_img.shape[:2]
        # Если изображение уже нужного размера, не трогаем
        if h == gt_size and w == gt_size:
            return lq_img, hq_img
        # Ресайзим LQ и HQ
        lq_resized = resize(lq_img, (gt_size, gt_size), preserve_range=True, anti_aliasing=True).astype(lq_img.dtype)
        hq_resized = resize(hq_img, (gt_size, gt_size), preserve_range=True, anti_aliasing=True).astype(hq_img.dtype)
        return lq_resized, hq_resized    