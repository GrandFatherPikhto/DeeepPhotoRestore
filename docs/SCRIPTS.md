# Скрипты и конфигурационные файлы проекта

## Конфигурация обучения

```yaml
# Общие настройки
name: NAFNet-JPEG-4070Ti
model_type: ImageRestorationModel
scale: 1
num_gpu: 1  # Используем твою одну видеокарту
manual_seed: 42

path:
  pretrain_network_g: ~    # Здесь можно будет указать путь к готовым весам для дообучения
  strict_load_g: true
  resume_state: ~          # Для продолжения обучения, если оно прервалось


# Настройка датасетов (Шаг 4.c из ТЗ)
datasets:
  train:
    name: MyTrainDataset
    type: PairedImageDataset
    # Пути к папкам, которые создал наш скрипт prepare_dataset.py
    dataroot_gt: ../dataset_ready/train/hq_targets  
    dataroot_lq: ../dataset_ready/train/lq_inputs   
    
    filename_tmpl: '{}'
    io_backend:
      type: disk

    # Настройки загрузчика данных
    gt_size: 256  # Нейросеть будет нарезать картинки на патчи 256x256 для обучения
    use_flip: true
    use_rot: true

    num_worker_per_gpu: 4
    batch_size_per_gpu: 8  # 8 картинок за один шаг (для 12ГБ памяти 4070Ti это ок)
    dataset_enlarge_ratio: 1

  val:
    name: MyTestDataset
    type: PairedImageDataset
    dataroot_gt: ../dataset_ready/test/hq_targets
    dataroot_lq: ../dataset_ready/test/lq_inputs
    io_backend:
      type: disk

# Настройки самой нейросети NAFNet
network_g:
  type: NAFNet
  width: 32
  enc_blk_nums: [2, 2, 4, 8]
  middle_blk_num: 12          # Переименовали blks: 12 вот в это имя
  dec_blk_nums: [2, 2, 2, 2]

# Параметры оптимизатора и обучения
train:
  optim_g:
    type: AdamW
    lr: !!float 1e-3
    weight_decay: 0
    betas: [0.9, 0.9]

  scheduler:
    type: TrueCosineAnnealingLR
    T_max: 200000
    eta_min: !!float 1e-7

  total_iter: 200000  # Сколько всего шагов будет учиться сеть
  warmup_iter: -1

  # Настройки потерь (метрики качества)
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

# Настройки сохранения прогресса и тестов прямо во время учебы
logger:
  print_freq: 200
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true

  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 0
      test_y_channel: false
```

## Скрипт для вытягивания картинок

Качаем с http://loki.disi.unitn.it/RAISE/confirm.php

Получае RAISE_v3.csv файл, в котором содержаться линки на загрузку и характеристики снимков

```python
import os
import pandas as pd
import requests
import rawpy
from PIL import Image
import io

# 1. Настройки
CSV_FILE = "RAISE_3.csv"  # Имя твоего CSV-файла
OUTPUT_DIR = "source_images"     # Папка, куда полетят готовые чистые картинки
MAX_IMAGES = 10                  # Для первого теста скачаем 10 штук

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Читаем CSV
print("Читаем CSV файл метаданных...")
df = pd.read_csv(CSV_FILE)

# 3. Качаем и конвертируем
downloaded = 0

for index, row in df.iterrows():
    if downloaded >= MAX_IMAGES:
        break
        
    # Пытаемся взять ссылку на TIFF (проще читать) или на NEF (RAW)
    url_tiff = row.get('TIFF')
    url_nef = row.get('NEF')
    
    # Предпочтение отдаем TIFF, если есть, так как его не надо декодировать из RAW
    url = url_tiff if (pd.notna(url_tiff)) else url_nef
    
    if not url or pd.isna(url):
        continue
        
    base_name = row.get('File')
    dest_jpg_path = os.path.join(OUTPUT_DIR, f"{base_name}.jpg")
    
    # Если уже скачивали — пропускаем
    if os.path.exists(dest_jpg_path):
        print(f"Файл {base_name} уже существует, пропускаю...")
        downloaded += 1
        continue

    print(f"[{downloaded+1}/{MAX_IMAGES}] Качаю: {base_name}...")
    
    try:
        response = requests.get(url, timeout=45)
        if response.status_code != 200:
            print(f"❌ Ошибка загрузки {base_name} (Код: {response.status_code})")
            continue
            
        # Если скачали RAW (.NEF), его нужно проявить в RGB
        if url.lower().endswith('.nef'):
            print(f"⚙️ Проявляю RAW (.NEF) для {base_name}...")
            with rawpy.imread(io.BytesIO(response.content)) as raw:
                # postprocess() превращает матрицу RAW в привычный RGB массив
                rgb_array = raw.postprocess(use_camera_wb=True) 
                img = Image.fromarray(rgb_array)
        else:
            # Если это TIFF, просто открываем его через Pillow
            img = Image.open(io.BytesIO(response.content))
            
        # Сохраняем в папку как качественный эталонный JPEG (100% качество)
        img.save(dest_jpg_path, "JPEG", quality=100)
        print(f"✅ Успешно сохранен: {dest_jpg_path}")
        downloaded += 1
        
    except Exception as e:
        print(f"❌ Сбой при обработке {base_name}: {e}")

print(f"\n🎉 Скрипт отработал! В папке '{OUTPUT_DIR}' теперь лежат готовые чистые файлы.")

```

## Скрипт подготовки датасетов для снимков

```python
import os
import shutil
import random
from PIL import Image

# 1. Настройка путей
SOURCE_DIR = "source_images"  # Сюда ты складываешь исходные чистые фото
BASE_OUT_DIR = "dataset_ready"

TRAIN_HQ = os.path.join(BASE_OUT_DIR, "train/hq_targets") # Оригиналы для обучения
TRAIN_LQ = os.path.join(BASE_OUT_DIR, "train/lq_inputs")  # Сжатые для обучения
TEST_HQ = os.path.join(BASE_OUT_DIR, "test/hq_targets")   # Оригиналы для теста
TEST_LQ = os.path.join(BASE_OUT_DIR, "test/lq_inputs")    # Сжатые для теста

# Автоматически создаем всю структуру папок
for folder in [TRAIN_HQ, TRAIN_LQ, TEST_HQ, TEST_LQ]:
    os.makedirs(folder, exist_ok=True)

# 2. Функция сжатия (Шаг 2 из ТЗ)
def compress_image(src_path, dest_path, quality_level=15):
    """Открывает картинку и пересохраняет с жестким JPEG-сжатием"""
    with Image.open(src_path) as img:
        # Конвертируем в RGB на случай, если попался PNG с прозрачностью (RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Сохраняем с низким качеством, порождая артефакты
        img.save(dest_path, "JPEG", quality=quality_level)

# 3. Чтение и перемешивание файлов
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
all_images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(valid_extensions)]

if not all_images:
    print(f"❌ Ошибка: Папка '{SOURCE_DIR}' пуста! Добавь туда исходные картинки.")
    exit()

random.seed(42)  # Фиксируем сид, чтобы разделение всегда было одинаковым
random.shuffle(all_images)

# Высчитываем пропорцию 95% / 5% (Шаг 4.c из ТЗ)
split_idx = int(len(all_images) * 0.95)
# Если картинок слишком мало для теста, отдаем хотя бы одну в тест
if split_idx == len(all_images) and len(all_images) > 1:
    split_idx -= 1

train_files = all_images[:split_idx]
test_files = all_images[split_idx:]

# 4. Функция распределения по папкам
def process_images(file_list, hq_dir, lq_dir, split_name):
    print(f"Обработка выборки [{split_name}] — {len(file_list)} шт...")
    for file_name in file_list:
        src_file_path = os.path.join(SOURCE_DIR, file_name)
        
        # Переносим оригинал (HQ Target) в неизменном виде (но в формате .png для стабильности нейросети)
        base_name = os.path.splitext(file_name)[0]
        hq_file_path = os.path.join(hq_dir, f"{base_name}.png")
        
        with Image.open(src_file_path) as img:
            img.save(hq_file_path, "PNG")
        
        # Создаем сжатый дубликат (LQ Input) в формате .jpg
        lq_file_path = os.path.join(lq_dir, f"{base_name}.jpg")
        compress_image(src_file_path, lq_file_path, quality_level=15)

# Запуск обработки
process_images(train_files, TRAIN_HQ, TRAIN_LQ, "TRAIN (95%)")
process_images(test_files, TEST_HQ, TEST_LQ, "TEST (5%)")

print("\n🎉 Всё готово! Проверь папку 'dataset_ready'. Она полностью укомплектована.")

```

## Скрипт для сравнения восстановленных картинок и оригинала

```python
import os
import torch
import numpy as np
from PIL import Image
from basicsr.models.archs.nafnet_arch import NAFNet

# ==================== НАСТРОЙКИ ====================
# 1. Путь к файлу весов модели (выбери самый свежий или latest)
MODEL_PATH = "NAFNet/experiments/NAFNet-JPEG-4070Ti/models/net_g_latest.pth"

# 2. Картинка, которую ты хочешь исправить (положи её в папку Tanahen)
INPUT_IMAGE = "my_shakal_photo.jpg" 

# 3. Куда сохранить результат работы нейросети
OUTPUT_IMAGE = "fixed_by_nafnet.png"
# ===================================================

def main():
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Ошибка: Положи зашакаленную картинку '{INPUT_IMAGE}' в папку проекта!")
        return

    # Настраиваем видеокарту 4070Ti
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Запускаем нейросеть на: {device}")

    # 1. Собираем пустую архитектуру NAFNet (точно такую же, как в конфиге)
    model = NAFNet(
        img_channel=3, 
        width=32, 
        middle_blk_num=12, 
        enc_blk_nums=[2, 2, 4, 8], 
        dec_blk_nums=[2, 2, 2, 2]
    )
    
    # 2. Загружаем в неё обученные веса из файла .pth
    print("🧠 Загружаем мозги нейросети...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # В BasicSR веса лежат внутри словаря под ключом 'params'
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval() # Переводим модель в режим предсказания

    # 3. Подготавливаем картинку для нейросети
    print("📸 Читаем и подготавливаем изображение...")
    img_pil = Image.open(INPUT_IMAGE).convert('RGB')
    
    # Превращаем картинку в массив чисел от 0.0 до 1.0
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    
    # Меняем порядок осей с (Высота, Ширина, Цвета) на (Цвета, Высота, Ширина) — так любит PyTorch
    img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float()
    
    # Добавляем фейковую ось "батча" в начало, чтобы получился тензор размера (1, 3, H, W)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 4. Сама магия ИИ
    print("🪄 Нейросеть убирает артефакты сжатия...")
    with torch.no_grad(): # Отключаем подсчет градиентов, чтобы видеокарта не тратила память
        output_tensor = model(img_tensor)

    # 5. Превращаем тензор обратно в обычную картинку
    output_tensor = output_tensor.squeeze(0).clamp(0, 1).cpu() # Убираем ось батча и обрезаем значения от 0 до 1
    output_np = np.transpose(output_tensor.numpy(), (1, 2, 0)) # Возвращаем оси (H, W, C)
    output_np = (output_np * 255.0).astype(np.uint8) # Переводим обратно в формат пикселей 0-255

    # Сохраняем результат
    fixed_img = Image.fromarray(output_np)
    fixed_img.save(OUTPUT_IMAGE)
    print(f"🎉 Готово! Результат сохранен в файл: {OUTPUT_IMAGE}")

if __name__ == '__main__':
    main()
```