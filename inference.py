#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import tifffile
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from libraries.pipeline_config import get_pipeline_config
from libraries.pipeline_device import get_torch_device
from libraries.pipeline_model import create_nafnet_model
from libraries.training_checkpoint import load_checkpoint
from libraries.pipeline_logger import setup_logger, get_logger
from libraries.pipeline_generation_core import generate_lq_from_hq

def load_lq_image(path, target_size=256):
    """
    Загружает 4-канальный TIFF и приводит к размеру target_size x target_size.
    Возвращает тензор [1, 4, H, W] float32 в диапазоне [0,1].
    """
    # Читаем как float32, нормализуем, если uint16
    img = tifffile.imread(path).astype(np.float32)
    if img.max() > 1.0:
        img = img / 65535.0
    # Приводим к размеру (target_size, target_size)
    h, w = img.shape[:2]
    if h != target_size or w != target_size:
        img = TF.resize(torch.from_numpy(img.transpose(2,0,1)), (target_size, target_size)).numpy().transpose(1,2,0)
    # Тензор [C, H, W]
    tensor = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)  # [1,4,H,W]
    return tensor

def load_or_degrade_input(path, config, target_size=256):
    # Проверяем, не TIFF ли с 4 каналами?
    if path.lower().endswith('.tiff') or path.lower().endswith('.tif'):
        try:
            img = tifffile.imread(path)
            if img.ndim == 3 and img.shape[2] == 4:
                # 4-канальный TIFF, считаем готовым LQ
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 65535.0
                # Ресайз до target_size
                h, w = img.shape[:2]
                if h != target_size or w != target_size:
                    img = TF.resize(torch.from_numpy(img.transpose(2,0,1)), (target_size, target_size)).numpy().transpose(1,2,0)
                tensor = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)
                return tensor, None  # GT нет
        except:
            pass
    # Иначе пробуем как RGB-изображение
    rgb = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    # Применяем деградацию (здесь нужно использовать те же параметры, что в обучении)
    lq_packed, hq_target = generate_lq_from_hq(rgb, config)   # lq_packed форма (H, W, 4)
    # Ресайз до target_size
    h, w = lq_packed.shape[:2]
    if h != target_size or w != target_size:
        lq_packed = TF.resize(torch.from_numpy(lq_packed.transpose(2,0,1)), (target_size, target_size)).numpy().transpose(1,2,0)
    tensor = torch.from_numpy(lq_packed.transpose(2,0,1)).float().unsqueeze(0)
    return tensor, hq_target   # GT может быть использован для сравнения (опционально)

def main():
    parser = argparse.ArgumentParser(description="Inference with trained NAFNet")
    parser.add_argument('-opt', required=True, help='Путь к конфигу (YAML)')
    parser.add_argument('--input', required=True, help='Путь к входному 4-канальному TIFF')
    parser.add_argument('--output', default='restored.png', help='Путь для сохранения результата (PNG)')
    parser.add_argument('--checkpoint', default=None, help='Путь к чекпоинту (если не указан, берётся из конфига)')
    args = parser.parse_args()

    # Настройка логгера
    opt = get_pipeline_config(args.opt) if hasattr(args, 'opt') else get_pipeline_config()  # небольшая адаптация
    # На самом деле get_pipeline_config() читает args из sys.argv, но для чистоты передадим opt_path
    # Упростим: загрузим конфиг вручную
    import yaml
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)
    log_cfg = opt.get('pipeline_logger', {})
    log_file = log_cfg.get('log_file', 'pipeline.log')
    setup_logger(log_file)
    logger = get_logger()

    device = get_torch_device()
    logger.info(f"Используется устройство: {device}")

    # Создаём модель
    model = create_nafnet_model(opt, device)

    # Загружаем веса
    checkpoint_path = args.checkpoint or opt['path'].get('resume_path', 'checkpoints/resume.pth')
    if not os.path.exists(checkpoint_path):
        logger.error(f"Чекпоинт не найден: {checkpoint_path}")
        sys.exit(1)

    # load_checkpoint ожидает много аргументов, но нам нужны только веса. Загрузим вручную.
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    logger.info(f"Модель загружена из {checkpoint_path}")

    # Загружаем входное изображение
    input_tensor = load_lq_image(args.input, target_size=256).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict):
            output = output['out']

    # Преобразуем обратно в изображение
    out_img = output.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1,2,0)
    out_img = (out_img * 255).astype(np.uint8)
    Image.fromarray(out_img).save(args.output)
    logger.info(f"Результат сохранён в {args.output}")


if __name__ == '__main__':
    main()