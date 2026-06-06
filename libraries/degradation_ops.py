# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def create_psf_kernel(sigma, size=15):
    ax = np.linspace(-(size//2), size//2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def add_correlated_noise(img, snr_db, psf_kernel):
    signal_power = np.mean(img ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    white_noise = np.random.normal(0, np.sqrt(noise_power), img.shape)
    correlated_noise = cv2.filter2D(white_noise, -1, psf_kernel)
    return np.clip(img + correlated_noise, 0, None)

def add_uncorrelated_noise(img, snr_db):
    signal_power = np.mean(img ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), img.shape)
    return np.clip(img + noise, 0, None)

def apply_bayer_mask(rgb, pattern='RGGB'):
    h, w, _ = rgb.shape
    bayer = np.zeros((h, w), dtype=rgb.dtype)
    bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]   # R
    bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]   # G1
    bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]   # G2
    bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]   # B
    return bayer

def extract_bayer_subchannels(bayer_2d):
    h, w = bayer_2d.shape
    h = h - (h % 2)
    w = w - (w % 2)
    bayer_2d = bayer_2d[:h, :w]
    r = bayer_2d[0::2, 0::2]
    g1 = bayer_2d[0::2, 1::2]
    g2 = bayer_2d[1::2, 0::2]
    b = bayer_2d[1::2, 1::2]
    return np.stack([r, g1, g2, b], axis=2)

def add_uncorrelated_noise_float(img_float, snr_db):
    """
    Добавляет белый гауссовский шум к изображению в формате float [0,1].
    """
    signal_power = np.mean(img_float ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), img_float.shape)
    return np.clip(img_float + noise, 0.0, 1.0)

def add_correlated_noise_float(img_float, snr_db, psf_kernel):
    """
    Добавляет коррелированный (свёрнутый с PSF) шум к изображению float [0,1].
    """
    signal_power = np.mean(img_float ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    white_noise = np.random.normal(0, np.sqrt(noise_power), img_float.shape)
    correlated_noise = cv2.filter2D(white_noise, -1, psf_kernel)
    return np.clip(img_float + correlated_noise, 0.0, 1.0)

def apply_bayer_mask_float(rgb_float, pattern='RGGB'):
    """
    Преобразует полноцветное RGB float [0,1] в одноканальный Bayer массив float [0,1].
    """
    h, w, _ = rgb_float.shape
    bayer = np.zeros((h, w), dtype=np.float32)
    bayer[0::2, 0::2] = rgb_float[0::2, 0::2, 0]   # R
    bayer[0::2, 1::2] = rgb_float[0::2, 1::2, 1]   # G1
    bayer[1::2, 0::2] = rgb_float[1::2, 0::2, 1]   # G2
    bayer[1::2, 1::2] = rgb_float[1::2, 1::2, 2]   # B
    return bayer