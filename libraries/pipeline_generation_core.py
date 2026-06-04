# -*- coding: utf-8 -*-
import numpy as np
import cv2
from libraries.pipeline_degradation_ops import (
    create_psf_kernel, add_correlated_noise, add_uncorrelated_noise,
    apply_bayer_mask, extract_bayer_subchannels
)

def generate_lq_from_hq(hq_rgb, config):
    proc_cfg = config.get('process_data', {})
    downscale = proc_cfg.get('downscale_factor', 1)
    noise_cfg = proc_cfg.get('noise', {})
    add_noise = noise_cfg.get('add', False)
    snr_db = noise_cfg.get('snr_db', 30)
    correlated = noise_cfg.get('correlated', False)
    psf_sigma = noise_cfg.get('psf_sigma', 1.5)

    if downscale > 1:
        h, w = hq_rgb.shape[:2]
        new_h, new_w = h // downscale, w // downscale
        hq_small = cv2.resize(hq_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        hq_small = hq_rgb.copy()

    psf_kernel = create_psf_kernel(psf_sigma)
    hq_blurred = np.zeros_like(hq_small, dtype=np.float32)
    for c in range(3):
        hq_blurred[..., c] = cv2.filter2D(hq_small[..., c].astype(np.float32), -1, psf_kernel)
    hq_target = np.clip(hq_blurred, 0, 255).astype(np.uint8)

    bayer = apply_bayer_mask(hq_target, pattern='RGGB')
    if add_noise:
        if correlated:
            bayer = add_correlated_noise(bayer, snr_db, psf_kernel)
        else:
            bayer = add_uncorrelated_noise(bayer, snr_db)

    bayer_16bit = np.clip(bayer.astype(np.float32) / 255.0 * 65535.0, 0, 65535).astype(np.uint16)
    lq_packed = extract_bayer_subchannels(bayer_16bit)
    return lq_packed, hq_target