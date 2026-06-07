# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from libraries.degradation_ops import (
    create_psf_kernel, add_uncorrelated_noise_float, add_correlated_noise_float,
    apply_bayer_mask_float, extract_bayer_subchannels
)

def generate_lq_from_hq(hq_rgb, config):
    """
    Генерирует пару (lq_packed, hq_target) для обучения БЕЗ ДЕСТРУКТИВНОГО РЕЗАЙЗА.
    lq_packed: 4-канальный RGGB (H_lq, W_lq, 4) float32, масштаб пикселей сохранён.
    hq_target: резкое RGB (H_hq, W_hq, 3) uint8, выровненное по сетке.
    """
    proc_cfg = config.get('process_data', {})
    data_cfg = config.get('datasets', {}).get('train', {})
    
    # Извлекаем целевые размеры кропов из конфига
    gt_size = data_cfg.get('gt_size', 256)   # Размер HQ патча
    lq_size = data_cfg.get('lq_size', 128)   # Размер LQ патча
    upscale_factor = config.get('datasets', {}).get('train', {}).get('upscale_factor', 2)
    
    noise_cfg = proc_cfg.get('noise', {})
    add_noise = noise_cfg.get('add', False)
    snr_db = noise_cfg.get('snr_db', 30)
    correlated = noise_cfg.get('correlated', False)
    psf_sigma = noise_cfg.get('psf_sigma', 1.5)

    # 1. Честное вырезание центрального патча из исходного огромного кадра HQ
    # Это гарантирует, что мы работаем с оригинальным масштабом матрицы сенсора
    h_orig, w_orig = hq_rgb.shape[:2]
    if h_orig < gt_size or w_orig < gt_size:
        # Если исходная картинка внезапно меньше целевого патча, аккуратно расширяем её
        hq_rgb = cv2.copyMakeBorder(hq_rgb, 0, max(0, gt_size - h_orig), 0, max(0, gt_size - w_orig), cv2.BORDER_REFLECT)
        h_orig, w_orig = hq_rgb.shape[:2]
        
    top_hq = (h_orig - gt_size) // 2
    left_hq = (w_orig - gt_size) // 2
    
    # Гарантируем чётность координат для сохранения фазы Bayer-сетки (RGGB)
    top_hq = top_hq - (top_hq % 2)
    left_hq = left_hq - (left_hq % 2)
    
    hq_target = hq_rgb[top_hq:top_hq+gt_size, left_hq:left_hq+gt_size, :].copy()
    hq_target = np.clip(hq_target, 0, 255).astype(np.uint8)

    # 2. Переводим в float для физически корректных деградаций
    hq_float = hq_target.astype(np.float32) / 255.0

    # 3. Моделируем размытие оптической системы (PSF Блур)
    from libraries.degradation_ops import create_psf_kernel
    psf_kernel = create_psf_kernel(psf_sigma)
    hq_blurred = np.zeros_like(hq_float)
    for c in range(3):
        hq_blurred[..., c] = cv2.filter2D(hq_float[..., c], -1, psf_kernel)

    # 4. Накладываем честную маску Bayer мозаики (каждый пиксель получает свой цвет)
    from libraries.degradation_ops import apply_bayer_mask_float
    bayer_float = apply_bayer_mask_float(hq_blurred, pattern='RGGB')

    # 5. Добавляем шумы матрицы
    from libraries.degradation_ops import add_correlated_noise_float, add_uncorrelated_noise_float
    if add_noise:
        if correlated:
            bayer_float = add_correlated_noise_float(bayer_float, snr_db, psf_kernel)
        else:
            bayer_float = add_uncorrelated_noise_float(bayer_float, snr_db)

    # 6. Упаковываем 2D-мозаику в 4 подканала RGGB. 
    # Размерность падает ровно в 2 раза: была (gt_size, gt_size), стала (gt_size//2, gt_size//2, 4)
    from libraries.degradation_ops import extract_bayer_subchannels
    lq_packed = extract_bayer_subchannels(bayer_float) # Результат: (128, 128, 4) при gt_size=256
    
    return lq_packed, hq_target
