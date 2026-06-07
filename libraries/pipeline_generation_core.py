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
    Генерирует пару (lq_packed, hq_target) для обучения с ЧЕСТНЫМ КОМБИНИРОВАННЫМ СЖАТИЕМ (4х).
    Общий масштаб деградации: downscale_factor (оптический) * 2 (Bayer subchannels) = 4.
    """
    proc_cfg = config.get('process_data', {})
    data_cfg = config.get('datasets', {}).get('train', {})
    
    # Извлекаем параметры масштабирования
    downscale = proc_cfg.get('downscale_factor', 2) # Оптическое уменьшение (2)
    gt_size = data_cfg.get('gt_size', 512)          # Размер HQ таргет-патча (например, 512)
    
    noise_cfg = proc_cfg.get('noise', {})
    add_noise = noise_cfg.get('add', False)
    snr_db = noise_cfg.get('snr_db', 30)
    correlated = noise_cfg.get('correlated', False)
    psf_sigma = noise_cfg.get('psf_sigma', 1.5)

    # 1. Извлекаем из огромного оригинала честный HQ патч размера gt_size x gt_size
    h_orig, w_orig = hq_rgb.shape[:2]
    top_hq = (h_orig - gt_size) // 2
    left_hq = (w_orig - gt_size) // 2
    
    # Фазовое выравнивание по сетке (кратно 4, так как общее сжатие = 4)
    top_hq = top_hq - (top_hq % 4)
    left_hq = left_hq - (left_hq % 4)
    
    hq_target = hq_rgb[top_hq:top_hq+gt_size, left_hq:left_hq+gt_size, :].copy()
    hq_target = np.clip(hq_target, 0, 255).astype(np.uint8)

    # 2. Моделируем оптический даунскейл (уменьшение проекции на матрицу камеры)
    # Из 512x512 получаем hq_small размером 256x256
    if downscale > 1:
        new_size = (gt_size // downscale, gt_size // downscale)
        hq_small = cv2.resize(hq_target, new_size, interpolation=cv2.INTER_LINEAR)
    else:
        hq_small = hq_target.copy()

    hq_small_float = hq_small.astype(np.float32) / 255.0

    # 3. Моделируем размытие пятна рассеяния оптической системы (PSF Блур)
    from libraries.degradation_ops import create_psf_kernel
    psf_kernel = create_psf_kernel(psf_sigma)
    hq_blurred = np.zeros_like(hq_small_float)
    for c in range(3):
        hq_blurred[..., c] = cv2.filter2D(hq_small_float[..., c], -1, psf_kernel)

    # 4. Накладываем маску Байера на уменьшенное размытое изображение
    from libraries.degradation_ops import apply_bayer_mask_float
    bayer_float = apply_bayer_mask_float(hq_blurred, pattern='RGGB')

    # 5. Добавляем шумы кремния
    from libraries.degradation_ops import add_correlated_noise_float, add_uncorrelated_noise_float
    if add_noise:
        if correlated:
            bayer_float = add_correlated_noise_float(bayer_float, snr_db, psf_kernel)
        else:
            bayer_float = add_uncorrelated_noise_float(bayer_float, snr_db)

    # 6. Расскладываем Bayer-мозаику (256x256) на 4 подканала RGGB
    # Размерность падает еще в 2 раза: на выходе получаем lq_packed размером (128, 128, 4)
    from libraries.degradation_ops import extract_bayer_subchannels
    lq_packed = extract_bayer_subchannels(bayer_float) 
    
    # Проверка для ВАК: из HQ (512x512x3) получили LQ (128x128x4). Общий масштаб деградации = 4!
    return lq_packed, hq_target

