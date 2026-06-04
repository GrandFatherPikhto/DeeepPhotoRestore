#!/./.venv/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from libraries.pipeline_logger import get_logger

logger = get_logger()

def bilinear_demosaic_rggb(lq_4ch):
    """
    Быстрая билинейная демозаика 4-канального RGGB.
    Вход: numpy array формы (H, W, 4) в диапазоне [0,1]
    Выход: RGB изображение формы (2H, 2W, 3) в диапазоне [0,1]
    """
    from scipy.ndimage import zoom
    r = lq_4ch[:,:,0]
    g1 = lq_4ch[:,:,1]
    g2 = lq_4ch[:,:,2]
    b = lq_4ch[:,:,3]
    # Увеличиваем каждый канал в 2 раза (билинейная интерполяция)
    r_up = zoom(r, 2, order=1)
    g1_up = zoom(g1, 2, order=1)
    g2_up = zoom(g2, 2, order=1)
    b_up = zoom(b, 2, order=1)
    g_up = (g1_up + g2_up) / 2.0
    rgb = np.stack([r_up, g_up, b_up], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def clean_old_visuals(output_dir, filenames):
    """Удаление старых отрендеренных превью из целевой директории."""
    for fname in filenames:
        full_path = os.path.join(output_dir, fname)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                logger.info(f"Старый файл визуализации удален: {full_path}")
            except Exception as e:
                logger.warning(f"Не удалось удалить {full_path}: {e}")

def save_tensor_as_image(tensor, output_path):
    """
    Конвертирует PyTorch тензор в изображение и сохраняет.
    Поддерживает: 3-канальный RGB (C, H, W) и 4-канальный RGGB (C, H, W).
    Для 4-канального применяет билинейную демозаику.
    """
    img_np = tensor.detach().cpu().numpy()
    
    if img_np.ndim == 3:
        if img_np.shape[0] == 4:
            # 4-канальный RGGB: применяем демозаику
            img_np = img_np.transpose(1, 2, 0)   # (H, W, 4)
            img_rgb = bilinear_demosaic_rggb(img_np)  # (2H, 2W, 3) float [0,1]
            img_np = (img_rgb * 255.0).astype(np.uint8)
        else:
            # 3-канальный RGB: просто переставляем оси
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    else:
        # fallback: если вдруг 2D или другое
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, img_np)
    logger.info(f"Диагностическое превью сохранено в: {output_path}")

def run_visual_control(dataset, config):
    """Основная функция визуального верификационного контроля датасета."""
    logger.info("=== Запуск модуля визуального контроля данных ===")
    
    vis_cfg = config.get("visuals_logger", {})
    output_dir = vis_cfg.get("output_dir", "samples/debug_visuals")
    lq_name = vis_cfg.get("lq_preview_name", "debug_lq_preview.png")
    gt_name = vis_cfg.get("gt_reference_name", "debug_gt_reference.png")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if config.get("clean_visuals", False):
        clean_old_visuals(output_dir, [lq_name, gt_name])
        
    if len(dataset) == 0:
        logger.error("Ошибка контроля: Датасет пуст!")
        raise IndexError("Dataset is empty")
        
    test_idx = np.random.randint(0, len(dataset))
    lq_tensor, hq_tensor = dataset[test_idx]
    
    logger.info(f"Контрольный сэмпл #{test_idx} успешно извлечен из выборки.")
    
    lq_path = os.path.join(output_dir, lq_name)
    gt_path = os.path.join(output_dir, gt_name)
    
    save_tensor_as_image(lq_tensor, lq_path)
    save_tensor_as_image(hq_tensor, gt_path)
    
    logger.info("📸 Визуальный тест успешно завершен.")