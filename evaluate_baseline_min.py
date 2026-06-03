#!./.venv/bin/python3
import os
import cv2
import numpy as np
import tifffile
from PIL import Image

def calculate_psnr(img1, img2):
    """Вычисление точного PSNR между массивами numpy в диапазоне [0, 1]"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def main():
    # Наш сгенерированный датасет
    dataset_root = "datasets/nef_nafnet"
    lq_dir = os.path.join(dataset_root, "test/lq_inputs")
    gt_dir = os.path.join(dataset_root, "test/hq_targets")
    
    if not os.path.exists(lq_dir):
        print(f"❌ Тестовая папка {lq_dir} не найдена.")
        return

    files = [f for f in os.listdir(lq_dir) if f.endswith('.tiff')]
    print(f"🔍 Найдено {len(files)} тестовых кадров. Запуск оптимизированного Baseline-расчета...")

    bilinear_psnrs = []
    mhc_psnrs = []

    for fname in files:
        base_name = fname.replace('_bayer.tiff', '')
        
        # 1. Загружаем сохраненный 4-канальный TIFF [4, H, W]
        bayer_packed = tifffile.imread(os.path.join(lq_dir, fname)).astype(np.float32) / 65535.0
        
        if bayer_packed.shape[0] != 4 and bayer_packed.shape[2] == 4:
            bayer_packed = bayer_packed.transpose(2, 0, 1)
            
        # Извлекаем подканалы, которые генератор сохранил как чистые слои
        r  = bayer_packed[0]
        g1 = bayer_packed[1]
        g2 = bayer_packed[2]
        b  = bayer_packed[3]
        
        # 2. Загружаем Ground Truth RGB референс и приводим к 256x256
        gt_path = os.path.join(gt_dir, f"{base_name}.png")
        if not os.path.exists(gt_path): continue
        gt_rgb = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0
        gt_resized = cv2.resize(gt_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

        # 🎯 ЧЕСТНЫЙ BASELINE ДЛЯ ТЕКУЩЕЙ СТРУКТУРЫ ДАННЫХ:
        # Просто увеличиваем чистые слои до 256x256 стандартным билинейным апскейлом
        r_up = cv2.resize(r, (256, 256), interpolation=cv2.INTER_LINEAR)
        b_up = cv2.resize(b, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        g_mean = (g1 + g2) / 2.0
        g_up = cv2.resize(g_mean, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Собираем итоговое Baseline RGB
        baseline_rgb = np.stack([r_up, g_up, b_up], axis=2)
        baseline_rgb = np.clip(baseline_rgb, 0.0, 1.0)

        # Расчет метрики качества
        p_bilinear = calculate_psnr(baseline_rgb, gt_resized)
        bilinear_psnrs.append(p_bilinear)

    print("\n📊 === ИТОГОВЫЕ РЕЗУЛЬТАТЫ СРАВНЕНИЯ (256x256 PATCh BASELINE) ===")
    print(f"1. Билинейный метод (Bilinear ФНЧ)    : Средний PSNR = {np.mean(bilinear_psnrs):.2f} dB")
    print(f"2. Градиентный метод (Edge-Aware MHC) : Средний PSNR = {np.mean(mhc_psnrs):.2f} dB")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
