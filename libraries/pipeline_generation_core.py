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

# libraries/pipeline_generation_core.py

def generate_lq_from_hq(hq_rgb, config):
    """
    Генерирует пару (lq_packed, hq_target) для обучения.
    lq_packed: 4-канальный RGGB (H_small, W_small, 4)  uint16
    hq_target: резкое RGB (H_original, W_original, 3) uint8
    """
    proc_cfg = config.get('process_data', {})
    downscale = proc_cfg.get('downscale_factor', 1)
    noise_cfg = proc_cfg.get('noise', {})
    add_noise = noise_cfg.get('add', False)
    snr_db = noise_cfg.get('snr_db', 30)
    correlated = noise_cfg.get('correlated', False)
    psf_sigma = noise_cfg.get('psf_sigma', 1.5)

    # 1. Резкий таргет – исходное полноразмерное RGB (без изменений)
    hq_target = np.clip(hq_rgb, 0, 255).astype(np.uint8)   # (H, W, 3)

    # 2. Для генерации LQ уменьшаем разрешение (если нужен downscale)
    if downscale > 1:
        h, w = hq_rgb.shape[:2]
        new_h, new_w = h // downscale, w // downscale
        hq_small = cv2.resize(hq_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        hq_small = hq_rgb.copy()

    # 3. Переводим hq_small в float [0,1] для операций деградации
    hq_small_float = hq_small.astype(np.float32) / 255.0

    # 4. Применяем размытие (PSF)
    psf_kernel = create_psf_kernel(psf_sigma)
    hq_blurred_float = np.zeros_like(hq_small_float)
    for c in range(3):
        hq_blurred_float[..., c] = cv2.filter2D(hq_small_float[..., c], -1, psf_kernel)

    # 5. Применяем маску Байера (получаем одноканальный Bayer float)
    bayer_float = apply_bayer_mask_float(hq_blurred_float, pattern='RGGB')

    # 6. Добавляем шум (если нужно)
    if add_noise:
        if correlated:
            bayer_float = add_correlated_noise_float(bayer_float, snr_db, psf_kernel)
        else:
            bayer_float = add_uncorrelated_noise_float(bayer_float, snr_db)

    # 7. Преобразуем в 16-бит и упаковываем в 4 канала
    # bayer_16bit = np.clip(bayer_float * 65535.0, 0, 65535).astype(np.uint16)
    # lq_packed = extract_bayer_subchannels(bayer_16bit)   # (H_small, W_small, 4)
    lq_packed = extract_bayer_subchannels(bayer_float)   # (H_small/2, W_small/2, 4)

    return lq_packed, hq_target