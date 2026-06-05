# -*- coding: utf-8 -*-
"""
Модуль подготовки датасета: проверка, очистка, генерация через прямые вызовы.
"""

import os
import shutil
from libraries.logger import get_logger
from libraries.pipeline_generation_io import process_source_images

logger = get_logger()

def clean_dataset_folders(dataset_root):
    """Удаляет папки train и test внутри dataset_root."""
    train_path = os.path.join(dataset_root, "train")
    test_path = os.path.join(dataset_root, "test")
    for p in [train_path, test_path]:
        if os.path.exists(p):
            shutil.rmtree(p)
            logger.info(f"Очистка: удалена папка {p}")
    # Пересоздаём структуру
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    logger.info(f"Очистка датасета завершена, корень {dataset_root} сохранён")

def is_dataset_empty(dataset_root):
    """Проверяет, пуст ли датасет (отсутствуют lq_inputs в train)."""
    lq_dir = os.path.join(dataset_root, "train", "lq_inputs")
    if not os.path.exists(lq_dir):
        return True
    return len(os.listdir(lq_dir)) == 0

def ensure_dataset_ready(config, clean_dataset=False):
    """
    Гарантирует, что датасет готов к использованию.
    Если clean_dataset=True – сначала очищает train/test.
    Если датасет пуст – запускает генерацию напрямую.
    """
    dataset_root = config.get("dataset_root", "datasets/nef_nafnet")
    logger.info(f"Работа с датасетом, корень: {dataset_root}")

    if clean_dataset:
        logger.info("Флаг clean_dataset=True: очищаем папки train/test")
        clean_dataset_folders(dataset_root)

    if is_dataset_empty(dataset_root):
        logger.info("Датасет пуст или не найден. Запуск генерации...")
        # Устанавливаем флаг очистки для генерации (чтобы она не дублировала очистку)
        config['clean_generation'] = clean_dataset
        process_source_images(config)
        logger.info("Генерация датасета завершена")
    else:
        lq_dir = os.path.join(dataset_root, "train", "lq_inputs")
        files_count = len(os.listdir(lq_dir))
        logger.info(f"Датасет уже существует ({files_count} патчей). Генерация пропущена.")