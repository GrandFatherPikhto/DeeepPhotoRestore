#!./.venv/bin/python3
import os
import argparse
import random
import yaml
import numpy as np
import imageio.v3 as iio
from PIL import Image
import tifffile
import cv2  # 🎯 ДОБАВЛЕНО: Теперь OpenCV доступен внутри этого скрипта!
from tqdm import tqdm
from pathlib import Path

# Импортируем нашу новую чистую библиотеку
from libraries.degradation import (
    create_psf_kernel, add_correlated_noise, add_uncorrelated_noise, 
    apply_bayer_mask, extract_bayer_subchannels
)

class RawDatasetPreparator:
    def __init__(self, opt):
        self.opt = opt
        self.dataset_type = opt['datasets']['train']['type']
        
        path_opt = opt.get('path', {})
        raw_source_dir = path_opt.get('source_images_dir', 'source_images')
        self.source_dir = os.path.expanduser(raw_source_dir)        
        
        dataset_root = opt['path']['dataset_root']
        self.train_hq = os.path.join(dataset_root, "train/hq_targets")
        self.train_lq = os.path.join(dataset_root, "train/lq_inputs")
        self.test_hq = os.path.join(dataset_root, "test/hq_targets")
        self.test_lq = os.path.join(dataset_root, "test/lq_inputs")
        
        self.train_ratio = opt.get('train_ratio', 0.95)
        self.seed = opt.get('manual_seed', 42)
        self.verbose = opt.get('logger', {}).get('verbose', True)

        for d in [self.train_hq, self.train_lq, self.test_hq, self.test_lq]:
            os.makedirs(d, exist_ok=True)

        process_cfg = opt.get('process_data', {})
        self.downscale_factor = process_cfg.get('downscale_factor', 1)
        self.add_noise = process_cfg.get('noise', {}).get('add', False)
        self.snr_db = process_cfg.get('noise', {}).get('snr_db', 30)
        self.correlated = process_cfg.get('noise', {}).get('correlated', False)
        self.psf_sigma = process_cfg.get('noise', {}).get('psf_sigma', 1.5)
        self.psf_kernel = create_psf_kernel(self.psf_sigma) if self.correlated else None

    def get_source_files(self):
        extensions = ('.nef', '.cr2', '.dng', '.arw', '.jpg', '.jpeg', '.png', '.tiff', '.tif')
        files = [f for f in os.listdir(self.source_dir) if f.lower().endswith(extensions)]
        return [os.path.join(self.source_dir, f) for f in files]

    def extract_hq_rgb(self, raw_path):
        try:
            rgb = iio.imread(raw_path)
            if rgb.dtype != np.uint8:
                rgb = (rgb / 65535 * 255).astype(np.uint8)
            if len(rgb.shape) == 2:
                rgb = np.stack((rgb,)*3, axis=-1)
            elif rgb.shape[2] == 4:
                rgb = rgb[:,:,:3]
            return rgb
        except Exception as e:
            if self.verbose: print(f"Ошибка чтения RAW {raw_path}: {e}")
            return None

    def synthesize_lq(self, hq_rgb):
        """Синтез физически адекватного 4-канального 16-битного входа"""
        bayer = apply_bayer_mask(hq_rgb, pattern='RGGB')
        
        if self.downscale_factor > 1:
            new_h = bayer.shape[0] // self.downscale_factor
            new_w = bayer.shape[1] // self.downscale_factor
            bayer = cv2.resize(bayer, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
        if self.add_noise:
            bayer = add_correlated_noise(bayer, self.snr_db, self.psf_kernel) if self.correlated else add_uncorrelated_noise(bayer, self.snr_db)
                
        # 🎯 ИСПРАВЛЕНО: Растягиваем 8-битные значения в честные 16-бит (0-65535) для точности
        bayer_16bit = np.clip((bayer / 255.0 * 65535.0), 0, 65535).astype(np.uint16)
        
        # Сразу раскладываем мозаику в 4 подканала под структуру NAFNet
        return extract_bayer_subchannels(bayer_16bit)

    def save_pair(self, lq_packed, hq_img, name, subset='train'):
        hq_dir = self.train_hq if subset == 'train' else self.test_hq
        lq_dir = self.train_lq if subset == 'train' else self.test_lq
            
        Image.fromarray(hq_img).save(os.path.join(hq_dir, f"{name}.png"))
        # Сохраняем честный 4-канальный 16-битный TIFF пакет
        tifffile.imwrite(os.path.join(lq_dir, f"{name}_bayer.tiff"), lq_packed, photometric='minisblack')

    def run(self, clean=False):
        # === БЛОК ЧИСТОЙ ОЧИСТКИ (Без забивания диска ZIP-архивами!) ===
        if clean:
            import shutil
            print("\n🧹 [Очистка] Запущена полная очистка перед генерацией...")
            
            # 1. Стираем только сгенерированный ранее датасет
            dataset_root = self.opt['path']['dataset_root']
            if os.path.exists(dataset_root):
                print(f"  - Удаляю старый датасет: {dataset_root}")
                shutil.rmtree(dataset_root)
                
            # 2. Стираем результаты только ТЕКУЩЕГО эксперимента (если имя совпало)
            exp_name = self.opt.get('name', 'default_exp')
            exp_dir = os.path.join('experiments', exp_name)
            if os.path.exists(exp_dir):
                print(f"  - Удаляю старые результаты обучения текущего эксперимента: {exp_dir}")
                shutil.rmtree(exp_dir)
                
            # Пересоздаем пустые чистые папки под новые файлы
            for d in [self.train_hq, self.train_lq, self.test_hq, self.test_lq]:
                os.makedirs(d, exist_ok=True)
            print("✨ Очистка завершена! Переходим к генерации данных.\n")

        src_files = self.get_source_files()
        if not src_files: return
        random.seed(self.seed)
        random.shuffle(src_files)
        
        split_idx = int(len(src_files) * self.train_ratio)
        train_files, test_files = src_files[:split_idx], src_files[split_idx:]

        for f in tqdm(train_files, desc="Генерация Train"):
            hq = self.extract_hq_rgb(f)
            if hq is not None: self.save_pair(self.synthesize_lq(hq), hq, os.path.splitext(os.path.basename(f))[0], 'train')
                
        for f in tqdm(test_files, desc="Генерация Test"):
            hq = self.extract_hq_rgb(f)
            if hq is not None: self.save_pair(self.synthesize_lq(hq), hq, os.path.splitext(os.path.basename(f))[0], 'test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True)
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    
    with open(args.opt, 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)
        
    RawDatasetPreparator(opt).run(clean=args.clean)
