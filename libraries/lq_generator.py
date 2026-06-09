# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

# Подключаем корень и импортируем твой движок демозаики
sys.path.insert(0, str(Path(__file__).parent.parent))
from libraries.pipeline_visuals import bilinear_demosaic_rggb


def convert_single_lq(tiff_path: Path, output_png_path: Path):
    """Читает 4-канальный TIFF, делает демозаику и сохраняет PNG (uint8)."""
    # Загружаем LQ, нормализуем в float [0,1]
    lq = tifffile.imread(str(tiff_path)).astype(np.float32) / 65535.0  # (H, W, 4)

    # Билинейная демозаика -> RGB float [0,1]
    rgb = bilinear_demosaic_rggb(lq)  # (2H, 2W, 3)

    # Преобразуем в uint8
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    
    # Конвертируем RGB -> BGR для корректного сохранения в OpenCV
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_png_path), bgr)


def generate_previews_from_config(config):
    """
    Автоматически генерирует превью для всех найденных DSC номеров,
    если они отсутствуют в целевой папке lq_preview эксперимента.
    """
    # Исходная папка с 4-канальными тиффами в датасете (тестовая выборка)
    source_dir = Path(config.dataset_root_str) / "test" / "lq_inputs"
    
    # Целевая папка для превьюшек внутри эксперимента
    target_dir = Path(config.experiment_dir_str) / "lq_preview"
    target_dir.mkdir(parents=True, exist_ok=True)

    missing_previews = []
    for dsc_num in config.dsc_numbers:
        # Учитываем суффикс _bayer в имени исходного файла TIFF
        tiff_path = source_dir / f"DSC_{dsc_num}_bayer.tiff"
        if not tiff_path.exists():
            tiff_path = source_dir / f"DSC_{dsc_num}_bayer.tif"
            
        # На всякий случай проверяем и чистый вариант без суффикса
        if not tiff_path.exists():
            tiff_path = source_dir / f"DSC_{dsc_num}.tiff"
        if not tiff_path.exists():
            tiff_path = source_dir / f"DSC_{dsc_num}.tif"
            
        # Если исходника вообще нет на диске — это критично
        if not tiff_path.exists():
            raise FileNotFoundError(f"Исходный LQ файл для DSC_{dsc_num} не найден в {source_dir} (проверены варианты с _bayer и без)")
            
        # Целевой путь для png-превью (строго DSC_XXXX_bayer.png)
        png_path = target_dir / f"DSC_{dsc_num}_bayer.png"
        
        # Если превьюшки ещё нет, добавляем в очередь на генерацию
        if not png_path.exists():
            missing_previews.append((tiff_path, png_path))

    if not missing_previews:
        print("Все необходимые LQ-превью уже сгенерированы.")
        return

    print(f"\nНайдено {len(missing_previews)} недостающих LQ-превью. Запускаем генерацию...")
    for tiff_p, png_p in tqdm(missing_previews, desc="Генерация LQ превью"):
        try:
            convert_single_lq(tiff_p, png_p)
        except Exception as e:
            print(f"Ошибка при обработке {tiff_p.name}: {e}")

