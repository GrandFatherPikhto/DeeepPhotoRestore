# -*- coding: utf-8 -*-
"""
Модуль управления датасетом: проверка, очистка, генерация через prepare_dataset.py.
"""

import os
import subprocess
import sys
import shutil

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from libraries.logger import get_logger

logger = get_logger()

def clean_dataset_folders(dataset_root):
    """Удаляет папки train и test внутри dataset_root, оставляя корень."""
    train_path = os.path.join(dataset_root, "train")
    test_path = os.path.join(dataset_root, "test")
    for p in [train_path, test_path]:
        if os.path.exists(p):
            shutil.rmtree(p)
            logger.info(f"Очистка: удалена папка {p}")
    # Пересоздаём структуру (чтобы не было ошибок при последующей записи)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    logger.info(f"Очистка датасета завершена, корень {dataset_root} сохранён")

def is_dataset_empty(dataset_root):
    """Проверяет, пуст ли датасет (отсутствуют lq_inputs в train)."""
    lq_dir = os.path.join(dataset_root, "train", "lq_inputs")
    if not os.path.exists(lq_dir):
        return True
    # Папка существует, но может быть пустой
    return len(os.listdir(lq_dir)) == 0

def ensure_dataset_ready(config, opt_path, clean_dataset=False):
    """
    Гарантирует, что датасет готов к использованию.
    Если clean_dataset=True – сначала очищает train/test.
    Если датасет пуст – запускает prepare_dataset.py.
    """
    path = config.get('path', None)
    if path is None:
        sys.exit(0)

    dataset_root = path.get("dataset_root", "datasets/nef_nafnet")
    logger.info(f"Работа с датасетом, корень: {dataset_root}")

    if clean_dataset:
        logger.info("Флаг clean_dataset=True: очищаем папки train/test")
        clean_dataset_folders(dataset_root)

    if is_dataset_empty(dataset_root):
        logger.info("Датасет пуст или не найден. Запуск prepare_dataset.py...")
        # Формируем команду
        cmd = [sys.executable, "prepare_dataset.py", "-opt", opt_path]
        # Если нужен флаг очистки для prepare_dataset (например --clean-dataset) – можно добавить
        # Пока передаём только -opt. При необходимости доработаем позже.
        try:
            subprocess.run(cmd, check=True)
            logger.info("prepare_dataset.py выполнен успешно")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при запуске prepare_dataset.py: {e}")
            raise
    else:
        lq_dir = os.path.join(dataset_root, "train", "lq_inputs")
        files_count = len(os.listdir(lq_dir))
        logger.info(f"Датасет уже существует ({files_count} патчей). Генерация пропущена.")

# def create_restoration_dataset(config, is_train=True):
#     """Создаёт и возвращает экземпляр RestorationDataset."""
#     from libraries.old.pipeline_data import RestorationDataset
#     return RestorationDataset(config, is_train=is_train)