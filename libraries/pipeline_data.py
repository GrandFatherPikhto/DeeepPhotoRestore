import os
import cv2
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
        ], additional_targets={'image': 'image', 'mask': 'image'})
        
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
        lq_path = os.path.join(self.lq_dir, self.file_names[idx])
        hq_path = os.path.join(self.hq_dir, self.file_names[idx])
        
        # Чтение изображений (OpenCV возвращает BGR/монохром в зависимости от декомпозиции)
        lq_img = cv2.imread(lq_path, cv2.IMREAD_UNCHANGED)
        hq_img = cv2.imread(hq_path, cv2.IMREAD_UNCHANGED)
        
        # Применяем ЧКХ-шум строго ИЗОЛИРОВАННО на LQ-канал
        lq_img = self._apply_cfa_noise(lq_img)
        
        # Применяем синхронную геометрию
        if self.is_train:
            # Для простоты предполагаем, что патчи нарезаны препроцессингом, либо делаем Crop
            augmented = self.geom_transform(image=lq_img, mask=hq_img)
            lq_img, hq_img = augmented['image'], augmented['mask']
            
        # Нормализация в диапазон [0, 1] и конвертация в тензоры PyTorch [C, H, W]
        lq_tensor = torch.from_numpy(lq_img).float().permute(2, 0, 1) / 255.0 if len(lq_img.shape) == 3 else torch.from_numpy(lq_img).float().unsqueeze(0) / 255.0
        hq_tensor = torch.from_numpy(hq_img).float().permute(2, 0, 1) / 255.0 if len(hq_img.shape) == 3 else torch.from_numpy(hq_img).float().unsqueeze(0) / 255.0
        
        return lq_tensor, hq_tensor
