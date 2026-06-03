#!./.venv/bin/python
import os
import sys
import argparse
import yaml
import importlib.util
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import tifffile
import cv2
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from custom_data_loaders.model_loader import load_network

# --- ПАРСЕР АРГУМЕНТОВ ---
parser = argparse.ArgumentParser(description="Скрипт восстановления изображений по конфигурационному файлу")
parser.add_argument('-opt', type=str, required=True, help='Путь к конфигурационному файлу .yml')
parser.add_argument('--source', type=str, default=None, help='Кастомный путь к файлу/папке (опционально)')
args = parser.parse_args()

# --- ЗАГРУЗКА НАСТРОЕК ИЗ YML ---
with open(args.opt, 'r', encoding='utf-8') as f:
    opt = yaml.safe_load(f)

exp_name = opt.get('name', 'default_experiment')
path_opt = opt.get('path', {})
checkpoint_path = path_opt.get('resume_path', 'checkpoints/resume_simple_jpeg.pth')

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"❌ Ошибка: Обученный чекпоинт не найден: {checkpoint_path}")

# --- 🎯 АВТОПОДБОР ПУТЕЙ ИЗ DATASET_ROOT (БЕЗ ПОВТОРОВ) ---
dataset_root = path_opt['dataset_root']

# Если пользователь не передал левый файл в консоли, берем стандартную тестовую папку lq_inputs
if args.source:
    source_path_str = args.source
else:
    source_path_str = os.path.join(dataset_root, "test", "lq_inputs")

# Картинки автоматически полетят в красивую изолированную папку эксперимента
out_dir_str = os.path.join('experiments', exp_name, 'restored_results')
os.makedirs(out_dir_str, exist_ok=True)

source_path = Path(source_path_str)

# --- ИНИЦИАЛИЗАЦИЯ И ДИНАМИЧЕСКИЙ ИМПОРТ МОДЕЛИ ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net_opt = opt.get('network_g', {})
model_type = net_opt.get('type', 'fcn_resnet50')
in_channels = net_opt.get('num_in_ch', 3)
out_channels = net_opt.get('num_out_ch', 3)

# Если в YAML прописан путь к внешней модели (например, к вашей NAFNet)
model = load_network(opt, device)

print(f"🔄 Загружаем обученные веса из {checkpoint_path}...")
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model')))
model.eval()

# --- СБОР ФАЙЛОВ ДЛЯ ОБРАБОТКИ ---
files_to_process = []
valid_extensions = ('.jpg', '.jpeg', '.png') if in_channels == 3 else ('.tiff', '.tif')

if source_path.is_file():
    if source_path.suffix.lower() in valid_extensions:
        files_to_process.append(source_path)
    else:
        print(f"❌ Ошибка: Файл {source_path.name} не соответствует режиму модели ({valid_extensions})")
        sys.exit(1)
elif source_path.is_dir():
    files_to_process = [p for p in source_path.iterdir() if p.suffix.lower() in valid_extensions]
else:
    print(f"❌ Ошибка: Указанный путь '{source_path_str}' не найден на диске.")
    sys.exit(1)

if not files_to_process:
    print(f"❌ Нет подходящих файлов ({valid_extensions}) по пути: {source_path_str}")
    sys.exit(1)

print(f"📂 Авто-источник (LQ): {source_path_str}")
print(f"📁 Папка назначения: {out_dir_str}")
print(f"🚀 Найдено файлов для восстановления: {len(files_to_process)} шт.")

# --- ФУНКЦИЯ РЕСТАВРАЦИИ ОДНОГО КАДРА ---
def restore_single_image(file_path):
    if in_channels == 3:
        input_img = Image.open(file_path).convert('RGB')
        orig_w, orig_h = input_img.size
        transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        input_tensor = transform(input_img).unsqueeze(0).to(device)
    else:
        bayer = tifffile.imread(str(file_path)).astype(np.float32) / 65535.0
        orig_h, orig_w = bayer.shape
        h, w = bayer.shape
        r = bayer[0:h:2, 0:w:2]
        g1 = bayer[0:h:2, 1:w:2]
        b = bayer[1:h:2, 0:w:2]
        g2 = bayer[1:h:2, 1:w:2]
        lq = np.stack([r, g1, b, g2], axis=2)
        lq_tensor = torch.from_numpy(lq.transpose(2, 0, 1)).float()
        lq_tensor = T.functional.resize(lq_tensor, (256, 256))
        input_tensor = lq_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict):
            output = output['out']

    output = output.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    output_resized = cv2.resize(output, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    final_img = (output_resized * 255.0).astype(np.uint8)
    
    out_name = f"restored_{file_path.stem.replace('_bayer', '')}.png"
    out_path = os.path.join(out_dir_str, out_name)
    Image.fromarray(final_img).save(out_path)

# --- ЗАПУСК ОБРАБОТКИ ---
for file_p in tqdm(files_to_process, desc="Реставрация"):
    try:
        restore_single_image(file_p)
    except Exception as e:
        print(f"\n⚠️ Ошибка при обработке {file_p.name}: {e}")

print(f"\n🎉 Все готово! Результаты сохранены в папку: {out_dir_str}")
