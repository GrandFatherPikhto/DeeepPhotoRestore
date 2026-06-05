# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import imageio.v3 as iio
import tifffile
from PIL import Image
from tqdm import tqdm
from libraries.pipeline_generation_core import generate_lq_from_hq
from libraries.logger import get_logger

logger = get_logger()

def load_rgb(filepath):
    try:
        img = iio.imread(filepath)
        if img.dtype != np.uint8:
            img = (img / 65535.0 * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    except Exception as e:
        logger.warning(f"Ошибка чтения {filepath}: {e}")
        return None

def process_source_images(config):
    path_opt = config.get('path', {})
    source_dir = os.path.expanduser(path_opt.get('source_images_dir', 'source_images'))
    dataset_root = path_opt.get('dataset_root', 'datasets/nef_nafnet')
    train_ratio = config.get('train_ratio', 0.95)
    seed = config.get('manual_seed', 42)
    clean = config.get('clean_generation', False)

    train_hq = os.path.join(dataset_root, 'train', 'hq_targets')
    train_lq = os.path.join(dataset_root, 'train', 'lq_inputs')
    test_hq = os.path.join(dataset_root, 'test', 'hq_targets')
    test_lq = os.path.join(dataset_root, 'test', 'lq_inputs')

    if clean:
        import shutil
        logger.info("Очистка датасета...")
        for p in [train_hq, train_lq, test_hq, test_lq]:
            if os.path.exists(p):
                shutil.rmtree(p)
    for p in [train_hq, train_lq, test_hq, test_lq]:
        os.makedirs(p, exist_ok=True)

    extensions = ('.nef', '.cr2', '.dng', '.arw', '.jpg', '.jpeg', '.png', '.tiff', '.tif')
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(extensions)]
    if not files:
        logger.error(f"Нет файлов в {source_dir}")
        return

    random.seed(seed)
    random.shuffle(files)
    split = int(len(files) * train_ratio)
    train_files, test_files = files[:split], files[split:]

    def process_file_list(file_list, subset):
        for fname in tqdm(file_list, desc=f"Генерация {subset}"):
            hq = load_rgb(os.path.join(source_dir, fname))
            if hq is None:
                continue
            lq_packed, hq_target = generate_lq_from_hq(hq, config)
            base = os.path.splitext(fname)[0]
            hq_dir = train_hq if subset == 'train' else test_hq
            lq_dir = train_lq if subset == 'train' else test_lq
            Image.fromarray(hq_target).save(os.path.join(hq_dir, f"{base}.png"))
            tifffile.imwrite(os.path.join(lq_dir, f"{base}_bayer.tiff"), lq_packed, photometric='minisblack')

    process_file_list(train_files, 'train')
    process_file_list(test_files, 'test')
    logger.info("Генерация завершена.")