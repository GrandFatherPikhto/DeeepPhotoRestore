#!./.venv/bin/python
# -*- coding: utf-8 -*-
"""
Скрипт для генерации пары (LQ, HQ) из одного RGB-изображения,
точно как в prepare_dataset.py (run_pipeline).

Использование:
    python generate_pair.py -opt config.yml --input image.png --output_dir ./output

Создаёт:
    - hq.png        (эталонный HQ-патч, центральный кроп размера gt_size)
    - lq.tiff       (упакованный LQ, 4 канала, uint16)
    - lq_preview.png (билинейная демозаика LQ для визуального контроля)
"""

import argparse
import os
import sys
import numpy as np
import tifffile
import cv2
import yaml
from PIL import Image
from pathlib import Path

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).parent))
from libraries.pipeline_generation_core import generate_lq_from_hq
from libraries.pipeline_visuals import bilinear_demosaic_rggb


def load_rgb_image(path):
    """Загружает RGB изображение в формате uint8 (0..255)"""
    img = np.array(Image.open(path).convert('RGB'))
    return img


def main():
    parser = argparse.ArgumentParser(description="Генерация пары (LQ, HQ) из одного изображения (как в run_pipeline).")
    parser.add_argument('-opt', required=True, help='Путь к YAML конфигу (например, options/train/NAFNet_100_02.yml)')
    parser.add_argument('--input', required=True, help='Входное RGB изображение (PNG, JPEG, TIFF)')
    parser.add_argument('--output_dir', default='./generated_pair', help='Папка для сохранения результатов')
    args = parser.parse_args()

    # Загружаем конфиг
    with open(args.opt, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Подстановка имени эксперимента (если нужно, но не критично)
    exp_name = config.get('name', '')
    if exp_name:
        # Простая замена {name} в путях конфига (для dataset_root и др.) – не требуется для этого скрипта
        pass

    # Загружаем исходное изображение
    hq_rgb = load_rgb_image(args.input)
    print(f"Загружено изображение: {hq_rgb.shape}")

    # Генерируем пару (LQ, HQ) точно как в prepare_dataset
    lq_packed, hq_target = generate_lq_from_hq(hq_rgb, config)

    # Создаём выходную папку
    os.makedirs(args.output_dir, exist_ok=True)

    # Сохраняем HQ (эталон)
    hq_path = os.path.join(args.output_dir, 'hq.png')
    Image.fromarray(hq_target).save(hq_path)
    print(f"HQ сохранён: {hq_path} (размер {hq_target.shape})")

    # Сохраняем LQ (4 канала, uint16)
    lq_16bit = np.clip(lq_packed * 65535.0, 0, 65535).astype(np.uint16)
    lq_path = os.path.join(args.output_dir, 'lq.tiff')
    tifffile.imwrite(lq_path, lq_16bit, photometric='minisblack')
    print(f"LQ сохранён: {lq_path} (размер {lq_packed.shape}, 4 канала)")

    # Визуализация LQ через билинейную демозаику (для просмотра)
    lq_demosaiced = bilinear_demosaic_rggb(lq_packed)  # (2H, 2W, 3) float [0,1]
    lq_preview = (lq_demosaiced * 255).astype(np.uint8)
    preview_path = os.path.join(args.output_dir, 'lq_preview.png')
    cv2.imwrite(preview_path, cv2.cvtColor(lq_preview, cv2.COLOR_RGB2BGR))
    print(f"Превью LQ (билинейная демозаика) сохранено: {preview_path}")

    print("Готово.")


if __name__ == '__main__':
    main()