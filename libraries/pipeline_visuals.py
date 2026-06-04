#!/./.venv/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from libraries.pipeline_logger import get_logger
logger = get_logger()

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
    """Конвертирует PyTorch тензор обратно в массив NumPy и сохраняет по указанному пути."""
    img_np = tensor.detach().cpu().numpy()
    if img_np.ndim == 3:
        if img_np.shape[0] == 4:  # 4-канальный RAW (RGGB)
            img_np = np.mean(img_np, axis=0)
        else:
            img_np = np.transpose(img_np, (1, 2, 0))
            
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, img_np)
    logger.info(f"Диагностическое превью сохранено в: {output_path}")

def run_visual_control(dataset, config):
    """Основная функция визуального верификационного контроля датасета."""
    logger.info("=== Запуск модуля визуального контроля данных ===")
    
    # Извлекаем параметры путей из YAML-конфига
    vis_cfg = config.get("visuals_logger", {})
    output_dir = vis_cfg.get("output_dir", "samples/debug_visuals")
    lq_name = vis_cfg.get("lq_preview_name", "debug_lq_preview.png")
    gt_name = vis_cfg.get("gt_reference_name", "debug_gt_reference.png")
    
    # Создаем директорию, если она отсутствует
    os.makedirs(output_dir, exist_ok=True)
    
    # Обработка флага полной очистки
    if config.get("clean_visuals", False):
        clean_old_visuals(output_dir, [lq_name, gt_name])
        
    if len(dataset) == 0:
        logger.error("Ошибка контроля: Датасет пуст!")
        raise IndexError("Dataset is empty")
        
    test_idx = np.random.randint(0, len(dataset))
    lq_tensor, hq_tensor = dataset[test_idx]
    
    logger.info(f"Контрольный сэмпл #{test_idx} успешно извлечен из выборки.")
    
    # Формируем финальные пути для записи
    lq_path = os.path.join(output_dir, lq_name)
    gt_path = os.path.join(output_dir, gt_name)
    
    save_tensor_as_image(lq_tensor, lq_path)
    save_tensor_as_image(hq_tensor, gt_path)
    
    logger.info("📸 Визуальный тест успешно завершен.")
