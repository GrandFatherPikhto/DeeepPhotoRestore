#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import tifffile
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from libraries.config import load_yaml_config
from libraries.device import get_torch_device
from libraries.logger import setup_logger, get_logger
from libraries.pipeline_visuals import bilinear_demosaic_rggb
from basicsr.models.archs.NAFNet_arch import NAFNet

def load_input_rgb(config):
    """Загружает входной TIFF, делает демозаику -> RGB. Возвращает (tensor, (H,W))."""
    input_path = config['paths']['input_tiff']
    bilinear = config.get('processing', {}).get('bilinear_demosaic', True)

    if bilinear:
        lq = tifffile.imread(input_path).astype(np.float32) / 65535.0
        rgb = bilinear_demosaic_rggb(lq)   # (2H, 2W, 3) float [0,1]
    else:
        # если подаётся уже RGB
        rgb = np.array(Image.open(input_path).convert('RGB')).astype(np.float32) / 255.0

    h, w = rgb.shape[:2]
    # Преобразуем в тензор [1,3,H,W]
    tensor = torch.from_numpy(rgb.transpose(2,0,1)).float().unsqueeze(0)
    return tensor, (h, w)

def load_gt(config, target_size=None):
    """Загружает эталонное RGB. Если target_size задан, ресайзит до этого размера."""
    gt_path = config['paths'].get('gt_png')
    if not gt_path or not os.path.exists(gt_path):
        return None
    img = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0
    if target_size is not None:
        h, w = img.shape[:2]
        if h != target_size or w != target_size:
            from torchvision.transforms import functional as TF
            img = TF.resize(torch.from_numpy(img.transpose(2,0,1)), target_size).numpy().transpose(1,2,0)
    tensor = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)
    return tensor

def create_model_from_config(config, device):
    net_cfg = config['network_g']
    model = NAFNet(
        img_channel=net_cfg['num_in_ch'],
        width=net_cfg['width'],
        middle_blk_num=net_cfg['middle_blk_num'],
        enc_blk_nums=net_cfg['enc_blk_nums'],
        dec_blk_nums=net_cfg['dec_blk_nums']
    )
    return model.to(device)

def load_pretrained_weights(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'params' in ckpt:
        state = ckpt['params']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def process_full_image(model, input_tensor, device):
    """Обрабатывает изображение целиком."""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict):
            output = output['out']
    return output.cpu()

def process_with_patches(model, input_tensor, device, patch_size=256, overlap=32):
    """
    Разбивает изображение на патчи с перекрытием, обрабатывает и собирает.
    Возвращает тензор [1,3,H,W].
    """
    _, _, H, W = input_tensor.shape
    # Шаг патча (без перекрытия)
    step = patch_size - overlap
    # Создаём массив для весов и накопления
    result = torch.zeros((1, 3, H, W), dtype=torch.float32)
    weight = torch.zeros((1, 1, H, W), dtype=torch.float32)

    for y in range(0, H, step):
        for x in range(0, W, step):
            # Координаты патча с учётом границ
            y1 = y
            y2 = min(y + patch_size, H)
            x1 = x
            x2 = min(x + patch_size, W)
            # Вырезаем патч
            patch = input_tensor[:, :, y1:y2, x1:x2].to(device)
            with torch.no_grad():
                out_patch = model(patch)
                if isinstance(out_patch, dict):
                    out_patch = out_patch['out']
            out_patch = out_patch.cpu()
            # Добавляем в результат с весами (простое перекрытие)
            result[:, :, y1:y2, x1:x2] += out_patch
            weight[:, :, y1:y2, x1:x2] += 1

    # Усредняем там, где было перекрытие
    result = result / weight
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.opt)

    # Настройка логгера
    log_cfg = config.get('pipeline_logger', {})
    log_file = log_cfg.get('log_file', 'pipeline.log')
    setup_logger(log_file)
    logger = get_logger()

    device = get_torch_device()
    logger.info(f"Устройство: {device}")

    # Создание модели
    model = create_model_from_config(config, device)
    ckpt_path = config['paths']['checkpoint']
    model = load_pretrained_weights(model, ckpt_path, device)
    logger.info(f"Модель загружена из {ckpt_path}")

    # Загрузка входного изображения
    input_tensor, orig_size = load_input_rgb(config)
    logger.info(f"Входное RGB размером {orig_size}")

    # Режим обработки
    proc_cfg = config.get('processing', {})
    mode = proc_cfg.get('mode', 'patch')
    if mode == 'full':
        output_tensor = process_full_image(model, input_tensor, device)
    else:
        patch_size = proc_cfg.get('patch_size', 256)
        overlap = proc_cfg.get('overlap', 32)
        logger.info(f"Режим патчей: patch_size={patch_size}, overlap={overlap}")
        output_tensor = process_with_patches(model, input_tensor, device, patch_size, overlap)

    # Сохранение результата
    out_img = output_tensor.squeeze(0).clamp(0,1).numpy().transpose(1,2,0)
    out_img = (out_img * 255).astype(np.uint8)

    output_dir = config['paths'].get('output_dir', '.')
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(config['paths']['input_tiff']))[0]
    out_path = os.path.join(output_dir, f"{base_name}_pretrained.png")
    Image.fromarray(out_img).save(out_path)
    logger.info(f"Результат сохранён в {out_path}")

    # Вычисление метрик (если есть GT)
    gt_tensor = load_gt(config, target_size=orig_size)  # GT тоже ресайзим под размер выхода
    if gt_tensor is not None:
        gt_np = gt_tensor.squeeze(0).cpu().numpy().transpose(1,2,0)
        psnr = peak_signal_noise_ratio(gt_np, out_img / 255.0, data_range=1)
        ssim = structural_similarity(gt_np, out_img / 255.0, channel_axis=2, data_range=1)
        logger.info(f"PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")

if __name__ == '__main__':
    main()