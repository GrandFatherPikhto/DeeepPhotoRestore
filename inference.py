#!./.venv/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import tifffile
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import yaml

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libraries.device import get_torch_device
from libraries.model_utils import create_nafnet_model
from libraries.pipeline_generation_core import generate_lq_from_hq
from libraries.logger import setup_logger, get_logger


def substitute_name(obj, name):
    """Рекурсивная замена {name} в строках."""
    if isinstance(obj, dict):
        return {k: substitute_name(v, name) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_name(i, name) for i in obj]
    elif isinstance(obj, str):
        return obj.replace('{name}', name)
    else:
        return obj


def load_input(path, config, target_size):
    """
    Загружает изображение и, если необходимо, применяет деградацию.
    Возвращает:
        lq_tensor: torch.Tensor [1, 4, target_size, target_size] (LQ)
        gt_tensor: torch.Tensor [1, 3, target_size, target_size] или None (если исходное RGB)
    """
    # 1. Пытаемся прочитать как готовый 4-канальный TIFF (упакованный LQ)
    if path.lower().endswith(('.tiff', '.tif')):
        try:
            img = tifffile.imread(path)
            if img.ndim == 3 and img.shape[2] == 4:
                # Уже LQ
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 65535.0
                h, w = img.shape[:2]
                if h != target_size or w != target_size:
                    img = TF.resize(torch.from_numpy(img.transpose(2, 0, 1)), (target_size, target_size)).numpy().transpose(1, 2, 0)
                lq_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
                return lq_tensor, None
        except Exception as e:
            print(f"Не удалось прочитать TIFF как 4-канальный: {e}")

    # 2. Иначе читаем как обычное RGB и применяем деградацию
    rgb = np.array(Image.open(path).convert('RGB'))  # uint8, [0, 255]
    lq_packed, hq_target = generate_lq_from_hq(rgb, config)  # lq_packed: (H, W, 4) float32 [0,1]; hq_target: (H, W, 3) uint8
    # Приводим к целевому размеру (если нужно)
    h, w = lq_packed.shape[:2]
    if h != target_size or w != target_size:
        lq_packed = TF.resize(torch.from_numpy(lq_packed.transpose(2, 0, 1)), (target_size, target_size)).numpy().transpose(1, 2, 0)
        # HQ тоже ресайзим, если он нужен для расчёта метрик (опционально)
        hq_target = (hq_target.astype(np.float32) / 255.0)   # временно переводим в float
        hq_target = TF.resize(torch.from_numpy(hq_target.transpose(2, 0, 1)), (target_size, target_size)).numpy().transpose(1, 2, 0)
        hq_target = (hq_target * 255).astype(np.uint8)
    else:
        hq_target = hq_target.astype(np.uint8)

    lq_tensor = torch.from_numpy(lq_packed.transpose(2, 0, 1)).float().unsqueeze(0)
    gt_tensor = torch.from_numpy(hq_target.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return lq_tensor, gt_tensor


def main():
    parser = argparse.ArgumentParser(description="Inference with trained NAFNet (JDSR)")
    parser.add_argument('-opt', required=True, help='Путь к конфигурационному YAML')
    parser.add_argument('--input', required=True, help='Входное изображение (RGB JPEG/PNG или 4-канальный TIFF)')
    parser.add_argument('--output', default='restored.png', help='Выходной PNG файл')
    parser.add_argument('--checkpoint', default=None, help='Путь к чекпоинту (опционально)')
    parser.add_argument('--target_size', type=int, default=None, help='Размер выходного квадратного изображения (по умолчанию gt_size из конфига)')
    args = parser.parse_args()

    # Загрузка конфига
    with open(args.opt, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Подстановка имени эксперимента (если есть)
    exp_name = config.get('name', '')
    if exp_name:
        config = substitute_name(config, exp_name)

    # Настройка логгера
    log_cfg = config.get('pipeline_logger', {})
    log_file = log_cfg.get('log_file', 'pipeline.log')
    setup_logger(log_file)
    logger = get_logger()

    device = get_torch_device()
    logger.info(f"Устройство: {device}")

    # Размер выходного изображения
    if args.target_size is not None:
        target_size = args.target_size
    else:
        target_size = config.get('datasets', {}).get('train', {}).get('gt_size', 256)
    logger.info(f"Целевой размер выходного изображения: {target_size}")

    # Создание модели
    model = create_nafnet_model(config, device)

    # Загрузка чекпоинта
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = config.get('path', {}).get('resume_path', 'checkpoints/resume.pth')
    if not os.path.exists(ckpt_path):
        logger.error(f"Чекпоинт не найден: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    logger.info(f"Модель загружена из {ckpt_path}")

    # Подготовка входного тензора
    lq_tensor, gt_tensor = load_input(args.input, config, target_size)
    lq_tensor = lq_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(lq_tensor)
        if isinstance(output, dict):
            output = output['out']

    # Сохранение результата
    out_img = output.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    out_img = (out_img * 255).astype(np.uint8)
    Image.fromarray(out_img).save(args.output)
    logger.info(f"Результат сохранён в {args.output}")

    # Если есть GT, считаем PSNR/SSIM
    if gt_tensor is not None:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        gt = gt_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        psnr = peak_signal_noise_ratio(gt, out_img / 255.0, data_range=1)
        ssim = structural_similarity(gt, out_img / 255.0, channel_axis=2, data_range=1)
        logger.info(f"PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")

if __name__ == '__main__':
    main()
