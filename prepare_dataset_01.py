#!./.venv/bin/python
import os
import sys
import argparse
import random
import yaml
import numpy as np
import imageio.v3 as iio
from PIL import Image
import tifffile
import cv2
from tqdm import tqdm
from pathlib import Path

# ----------------------------------------------
# Вспомогательные функции (Ваша математика)
# ----------------------------------------------

def create_psf_kernel(sigma, size=15):
    ax = np.linspace(-(size // 2), size // 2, size)
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
    bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]   # G
    bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]   # G
    bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]   # B
    return bayer

def compress_image(src_path, dest_path, quality_level=15):
    """Жесткое JPEG-сжатие для обычного режима"""
    with Image.open(src_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dest_path, "JPEG", quality=quality_level)

# ----------------------------------------------
# Основной класс подготовки датасета
# ----------------------------------------------

class RawDatasetPreparator:
    def __init__(self, opt):
        self.opt = opt
        
        # Считываем тип датасета из YAML конфигурации
        self.dataset_type = opt['datasets']['train']['type']
        
        # Динамически вытягиваем пути из структуры BasicSR YAML
        path_opt = opt.get('path', {})
        # self.source_dir = path_opt.get('source_images_dir', 'source_images')
        # 🎯 ДОБАВЛЕНО: Автоматически превращаем ~ в полный путь к домашней папке пользователя!
        raw_source_dir = path_opt.get('source_images_dir', 'source_images')
        # Если путь начинается с ~/, заменяем его на реальный домашний путь
        if raw_source_dir.startswith('~/'):
            self.source_dir = str(Path.home() / raw_source_dir[2:])
        else:
            self.source_dir = raw_source_dir

        self.source_dir = os.path.expanduser(raw_source_dir)        
        
        # 🎯 Боремся с повторами! Автоматически собираем структуру вокруг dataset_root
        dataset_root = opt['path']['dataset_root']
        self.train_hq = os.path.join(dataset_root, "train/hq_targets")
        self.train_lq = os.path.join(dataset_root, "train/lq_inputs")
        self.test_hq = os.path.join(dataset_root, "test/hq_targets")
        self.test_lq = os.path.join(dataset_root, "test/lq_inputs")
        
        # Базовые настройки
        self.train_ratio = opt.get('train_ratio', 0.95)
        self.seed = opt.get('manual_seed', 42)
        self.verbose = opt.get('logger', {}).get('verbose', True)

        # Создаём выходные папки на диске
        for d in [self.train_hq, self.train_lq, self.test_hq, self.test_lq]:
            os.makedirs(d, exist_ok=True)

        # Извлекаем параметры обработки данных из кастомного блока в YAML
        process_cfg = opt.get('process_data', {})
        self.downscale_factor = process_cfg.get('downscale_factor', 1)
        self.add_noise = process_cfg.get('noise', {}).get('add', False)
        self.snr_db = process_cfg.get('noise', {}).get('snr_db', 30)
        self.correlated = process_cfg.get('noise', {}).get('correlated', False)
        self.psf_sigma = process_cfg.get('noise', {}).get('psf_sigma', 1.5)
        self.psf_kernel = create_psf_kernel(self.psf_sigma) if self.correlated else None

    def get_source_files(self):
        # Смотрим, какой тип датасета выбран в YAML, чтобы определить расширения файлов
        if self.dataset_type in ['CustomRAWPairDataset', 'CustomNEFPairDataset']:
            extensions = ('.nef', '.cr2', '.dng', '.arw', '.raf', '.orf')
            print(f"🔍 Режим NEF: ищем файлы снимков камер в '{self.source_dir}'...")
        else:
            extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            print(f"🔍 Режим JPEG: ищем обычные изображения в '{self.source_dir}'...")
            
        files = [f for f in os.listdir(self.source_dir) if f.lower().endswith(extensions)]
        return [os.path.join(self.source_dir, f) for f in files]

    def extract_hq_rgb(self, raw_path):
        """Извлечение эталонного RGB с помощью imageio"""
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
            if self.verbose:
                print(f"Ошибка чтения RAW {raw_path}: {e}")
            return None

    def synthesize_lq(self, hq_rgb):
        """Синтез LQ из HQ (байеровская маска + downscale + шум) -> 16-бит"""
        bayer = apply_bayer_mask(hq_rgb, pattern='RGGB')
        if self.downscale_factor > 1:
            new_h = bayer.shape[0] // self.downscale_factor
            new_w = bayer.shape[1] // self.downscale_factor
            bayer = cv2.resize(bayer, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if self.add_noise:
            if self.correlated:
                bayer = add_correlated_noise(bayer, self.snr_db, self.psf_kernel)
            else:
                bayer = add_uncorrelated_noise(bayer, self.snr_db)
        bayer = np.clip(bayer, 0, 65535).astype(np.uint16)
        return bayer

    def save_pair(self, lq_img, hq_img, name, subset='train'):
        """Сохранение пары для режима NEF/RAW"""
        hq_dir = self.train_hq if subset == 'train' else self.test_hq
        lq_dir = self.train_lq if subset == 'train' else self.test_lq
            
        hq_path = os.path.join(hq_dir, f"{name}.png")
        Image.fromarray(hq_img).save(hq_path)
        
        lq_path = os.path.join(lq_dir, f"{name}_bayer.tiff")
        tifffile.imwrite(lq_path, lq_img, photometric='minisblack')

    def process_one_raw(self, raw_path, subset='train'):
        """Обработка одного кадра в режиме NEF/RAW"""
        base_name = os.path.splitext(os.path.basename(raw_path))[0]
        hq_rgb = self.extract_hq_rgb(raw_path)
        if hq_rgb is None or hq_rgb.shape[0] < 8 or hq_rgb.shape[1] < 8:
            return

        lq_bayer = self.synthesize_lq(hq_rgb)
        self.save_pair(lq_bayer, hq_rgb, base_name, subset)

    def process_one_jpeg(self, file_path, subset='train'):
        """Обработка одного файла в режиме JPEG (Никаких ложных тиффов!)"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        hq_dir = self.train_hq if subset == 'train' else self.test_hq
        lq_dir = self.train_lq if subset == 'train' else self.test_lq
        
        hq_path = os.path.join(hq_dir, f"{base_name}.png")
        lq_path = os.path.join(lq_dir, f"{base_name}.jpg")
        
        with Image.open(file_path) as img:
            img.save(hq_path, "PNG")
        compress_image(file_path, lq_path, quality_level=15)

    def run(self, clean=False):
        # === БЛОК ОЧИСТКИ И БЭКАПА ===
        if clean:
            import shutil
            from datetime import datetime
            print("\n🧹 [Очистка] Запущена полная очистка перед генерацией...")
            
            # 1. Очистка подготовленного датасета
            dataset_root = self.opt['path']['dataset_root']
            if os.path.exists(dataset_root):
                print(f"  - Удаляю старый датасет: {dataset_root}")
                shutil.rmtree(dataset_root)
                
            # 2. Бэкап и очистка папки экспериментов (логов, чекпоинтов, валидации)
            exp_name = self.opt.get('name', 'default_exp')
            exp_dir = os.path.join('experiments', exp_name)
            
            if os.path.exists(exp_dir):
                # Создаем папку для бэкапов, если её нет
                os.makedirs('backups', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_zip_path = os.path.join('backups', f"backup_{exp_name}_{timestamp}")
                
                print(f"  - Создаю бэкап старого эксперимента в: {backup_zip_path}.zip")
                # Архивируем папку эксперимента перед удалением
                shutil.make_archive(backup_zip_path, 'zip', exp_dir)
                
                print(f"  - Удаляю старые результаты обучения: {exp_dir}")
                shutil.rmtree(exp_dir)
                
            # Пересоздаем чистые папки на диске заново
            for d in [self.train_hq, self.train_lq, self.test_hq, self.test_lq]:
                os.makedirs(d, exist_ok=True)
            print("✨ Очистка и бэкап завершены! Переходим к генерации.\n")

        # === ДАЛЬШЕ ИДЕТ ВАШ СТАНДАРТНЫЙ КОД МЕТОДА RUN ===
        src_files = self.get_source_files() 
        if not src_files:
            print(f"❌ Файлы для обработки не найдены в папке: {self.source_dir}")
            return
        random.seed(self.seed)
        random.shuffle(src_files)
        
        split_idx = int(len(src_files) * self.train_ratio)
        if split_idx == len(src_files) and len(src_files) > 1:
            split_idx -= 1
            
        train_files = src_files[:split_idx]
        test_files = src_files[split_idx:]

        print(f"🚀 Старт генерации датасета из {self.source_dir} для режима '{self.dataset_type}'")
        print(f"Файлов всего: {len(src_files)} | В обучение: {len(train_files)} | В тест: {len(test_files)}")

        is_nef_mode = self.dataset_type in ['CustomRAWPairDataset', 'CustomNEFPairDataset']

        for f in tqdm(train_files, desc="Генерация Train"):
            if is_nef_mode:
                self.process_one_raw(f, 'train')
            else:
                self.process_one_jpeg(f, 'train')
                
        for f in tqdm(test_files, desc="Генерация Test"):
            if is_nef_mode:
                self.process_one_raw(f, 'test')
            else:
                self.process_one_jpeg(f, 'test')

        print(f"\n🎉 Подготовка датасета успешно завершена!")


# ----------------------------------------------
# Точка входа
# ----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help="Путь к конфигурационному файлу .yml")
    # Добавляем наш новый ключ очистки (action='store_true' делает его флагом: если он есть, то clean=True)
    parser.add_argument('--clean', action='store_true', help="Полная очистка датасета и бэкап старого обучения")
    args = parser.parse_args()
    
    with open(args.opt, 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)
        
    prep = RawDatasetPreparator(opt)
    # Передаем значение флага в метод run
    prep.run(clean=args.clean)
