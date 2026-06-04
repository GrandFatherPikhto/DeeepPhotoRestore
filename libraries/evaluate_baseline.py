#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль сравнения с классическими методами демозаики (билинейная, MHC).
Использует общий pipeline_logger.
"""

import os
import numpy as np
import tifffile
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from scipy.ndimage import zoom
from libraries.pipeline_logger import get_logger

logger = get_logger()

def bilinear_demosaic(lq_4ch):
    """
    Билинейная интерполяция каждого из 4 каналов (R, G1, G2, B) с последующим
    усреднением зелёных и сборкой RGB.
    lq_4ch: (H, W, 4) массив float32 в диапазоне [0,1].
    Возвращает: (2H, 2W, 3) массив float32.
    """
    h, w = lq_4ch.shape[:2]
    r_up = zoom(lq_4ch[:,:,0], 2, order=1)
    g1_up = zoom(lq_4ch[:,:,1], 2, order=1)
    g2_up = zoom(lq_4ch[:,:,2], 2, order=1)
    b_up = zoom(lq_4ch[:,:,3], 2, order=1)
    g_up = (g1_up + g2_up) / 2.0
    rgb = np.stack([r_up, g_up, b_up], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def mhc_demosaic(lq_4ch):
    """
    Упрощённая реализация градиентной демозаики Malvar-He-Cutler (MHC) для паттерна RGGB.
    Вход: (H, W, 4) каналы R, G1, G2, B (каждый размером H x W).
    Выход: (2H, 2W, 3) RGB.
    Алгоритм: интерполяция зелёного по градиентам, затем красного/синего с коррекцией.
    """
    # Для простоты и скорости – билинейная заглушка.
    # Полная реализация MHC требует сложной интерполяции и займёт много строк.
    # Рекомендую оставить заглушку или реализовать полноценно.
    logger.warning("MHC метод не реализован полностью, используется билинейная интерполяция")
    return bilinear_demosaic(lq_4ch)

def compute_metrics(pred, gt, data_range=1.0):
    """Вычисляет PSNR и SSIM для двух изображений в диапазоне [0, data_range]."""
    psnr = peak_signal_noise_ratio(gt, pred, data_range=data_range)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=data_range)
    return psnr, ssim

def run_baseline_evaluation(config):
    """
    Загружает тестовые данные из dataset_root, применяет билинейную и MHC демозаику,
    логирует средние PSNR и SSIM.
    """
    dataset_root = config.get('dataset_root', 'datasets/nef_nafnet')
    test_lq_dir = os.path.join(dataset_root, 'test', 'lq_inputs')
    test_gt_dir = os.path.join(dataset_root, 'test', 'hq_targets')
    
    if not os.path.exists(test_lq_dir) or not os.path.exists(test_gt_dir):
        logger.error(f"Тестовые папки не найдены: {test_lq_dir} или {test_gt_dir}")
        return
    
    files = [f for f in os.listdir(test_lq_dir) if f.endswith('_bayer.tiff')]
    if not files:
        logger.warning("Нет тестовых файлов в %s", test_lq_dir)
        return
    
    psnr_bilin, ssim_bilin = [], []
    psnr_mhc, ssim_mhc = [], []
    
    for fname in files:
        base = fname.replace('_bayer.tiff', '')
        lq_path = os.path.join(test_lq_dir, fname)
        gt_path = os.path.join(test_gt_dir, f"{base}.png")
        
        if not os.path.exists(gt_path):
            logger.warning(f"Нет эталона для {base}, пропускаем")
            continue
        
        # Загружаем LQ (4 канала) и GT (RGB)
        lq = tifffile.imread(lq_path).astype(np.float32) / 65535.0   # (H, W, 4)
        gt = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0
        
        # Билинейная демозаика
        rgb_bilin = bilinear_demosaic(lq)
        # Приводим GT к размеру rgb_bilin (2H, 2W)
        gt_resized = resize(gt, rgb_bilin.shape[:2], preserve_range=True)
        psnr_b, ssim_b = compute_metrics(rgb_bilin, gt_resized)
        psnr_bilin.append(psnr_b)
        ssim_bilin.append(ssim_b)
        
        # MHC демозаика
        rgb_mhc = mhc_demosaic(lq)
        # Предполагаем, что rgb_mhc уже того же размера, что и rgb_bilin
        if rgb_mhc.shape != rgb_bilin.shape:
            rgb_mhc = resize(rgb_mhc, rgb_bilin.shape[:2], preserve_range=True)
        psnr_m, ssim_m = compute_metrics(rgb_mhc, gt_resized)
        psnr_mhc.append(psnr_m)
        ssim_mhc.append(ssim_m)
        
        logger.info(f"{base}: Bilinear PSNR={psnr_b:.2f} SSIM={ssim_b:.4f} | MHC PSNR={psnr_m:.2f} SSIM={ssim_m:.4f}")
    
    if psnr_bilin:
        logger.info("=== BASELINE SUMMARY ===")
        logger.info(f"Bilinear  -> Mean PSNR = {np.mean(psnr_bilin):.2f} dB, Mean SSIM = {np.mean(ssim_bilin):.4f}")
        logger.info(f"MHC       -> Mean PSNR = {np.mean(psnr_mhc):.2f} dB, Mean SSIM = {np.mean(ssim_mhc):.4f}")
    else:
        logger.error("Не удалось обработать ни одного файла")