#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для применения гауссова размытия (PSF), шума и опционального уменьшения (downscale)
к RGB-изображению по настройкам из YAML-конфига (секция process_data).

Создаёт два выходных файла:
    --output_noisy   : шум + размытие в исходном размере
    --output_downscaled : шум + размытие после уменьшения (downscale_factor)
"""

import argparse
import sys
import numpy as np
import cv2
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from libraries.degradation_ops import create_psf_kernel, add_correlated_noise_float, add_uncorrelated_noise_float


def load_process_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    proc = config.get('process_data', {})
    downscale = proc.get('downscale_factor', 1)
    noise_cfg = proc.get('noise', {})
    return downscale, noise_cfg


def apply_blur_and_noise(img_float, noise_cfg, psf_kernel):
    """Применяет размытие и шум к float-изображению [0,1] (H,W,C)."""
    blurred = np.zeros_like(img_float)
    for c in range(3):
        blurred[..., c] = cv2.filter2D(img_float[..., c], -1, psf_kernel)

    add = noise_cfg.get('add', False)
    if add:
        snr_db = noise_cfg.get('snr_db', 30)
        correlated = noise_cfg.get('correlated', False)
        if correlated:
            noisy = np.zeros_like(blurred)
            for c in range(3):
                noisy[..., c] = add_correlated_noise_float(blurred[..., c], snr_db, psf_kernel)
        else:
            noisy = add_uncorrelated_noise_float(blurred, snr_db)
    else:
        noisy = blurred
    return noisy


def main():
    parser = argparse.ArgumentParser(description="Применить размытие, шум и downscale (опционально) к изображению.")
    parser.add_argument('-opt', required=True, help='Путь к YAML конфигу')
    parser.add_argument('--input', required=True, help='Входное RGB изображение')
    parser.add_argument('--output_noisy', default='degraded_noisy.png', help='Выходное (шум+размытие, исходный размер)')
    parser.add_argument('--output_downscaled', default='degraded_downscaled.png', help='Выходное после downscale')
    args = parser.parse_args()

    downscale, noise_cfg = load_process_config(args.opt)
    if not noise_cfg:
        print("В конфиге нет секции process_data.noise. Использую значения по умолчанию.")
        noise_cfg = {'add': True, 'snr_db': 30, 'correlated': False, 'psf_sigma': 1.5}

    # Чтение и конвертация в RGB float
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        print(f"Не удалось прочитать {args.input}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    h, w = img_float.shape[:2]

    # PSF ядро
    psf_sigma = noise_cfg.get('psf_sigma', 1.5)
    psf_kernel = create_psf_kernel(psf_sigma)

    # 1. Применяем размытие + шум в исходном размере
    noisy_full = apply_blur_and_noise(img_float, noise_cfg, psf_kernel)
    noisy_full_uint8 = (np.clip(noisy_full, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(args.output_noisy, cv2.cvtColor(noisy_full_uint8, cv2.COLOR_RGB2BGR))
    print(f"Сохранено (исходный размер): {args.output_noisy}")

    # 2. Если downscale > 1, применяем уменьшение, затем размытие+шум на уменьшенном
    if downscale > 1:
        new_w = w // downscale
        new_h = h // downscale
        img_down = cv2.resize(img_float, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        noisy_down = apply_blur_and_noise(img_down, noise_cfg, psf_kernel)
        noisy_down_uint8 = (np.clip(noisy_down, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(args.output_downscaled, cv2.cvtColor(noisy_down_uint8, cv2.COLOR_RGB2BGR))
        print(f"Сохранено (уменьшенное в {downscale} раз): {args.output_downscaled}")
    else:
        print("downscale_factor = 1, уменьшенная версия не создана.")

if __name__ == '__main__':
    main()