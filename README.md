# DeepPhotoRestore – Восстановление RAW‑изображений на основе NAFNet

**DeepPhotoRestore** – программный комплекс для совместной демозаики и суперразрешения (Joint Demosaicing and Super‑Resolution, JDSR) изображений, полученных с матрицей Байера (RGGB). Проект реализует сквозной конвейер: от физически корректной генерации синтетических пар (LQ – упакованный RAW, HQ – резкое RGB) до обучения глубокой нейросетевой модели на базе адаптированной архитектуры NAFNet с частотной функцией потерь (Focal Frequency Loss).

---

## 📌 Возможности

- **Генерация датасета** с моделированием реального тракта: оптическое уменьшение (`downscale_factor`), PSF‑размытие, маска Байера, коррелированный/некоррелированный шум.
- **Адаптированная модель NAFNet** с поддержкой 4‑канального входа (RGGB) и произвольного масштаба восстановления (`PixelShuffle`).
- **Комбинированная функция потерь**: L1 + Focal Frequency Loss (линейная или логарифмическая) с программируемым прогревом.
- **Полный цикл обучения** с автоматическим возобновлением (в т.ч. аварийным), клиппингом градиентов, валидацией (PSNR, SSIM) и логированием (CSV, TensorBoard).
- **Бенчмаркинг**: сравнение с классическими методами демозаики (билинейная, Malvar‑He‑Cutler).
- **Инференс на видео** (опционально).
- **Визуализация метрик** через скрипт построения графиков.

---

## 🗂️ Структура проекта

```text
.
├── configs/                     # Конфигурационные YAML‑файлы
├── experiments/                 # Результаты экспериментов (логи, чекпоинты, графики)
├── datasets/                    # Сгенерированные датасеты
├── source_images/               # Исходные резкие изображения (NEF, PNG, JPEG)
├── libraries/                   # Основной код
│   ├── config.py                # Загрузка конфигурации, обработка аргументов
│   ├── device.py                # Определение устройства (CUDA/CPU)
│   ├── logger.py                # Настройка логгера
│   ├── degradation_ops.py       # Операции деградации (PSF, шум, Байер)
│   ├── pipeline_generation_*.py # Генерация датасета
│   ├── training_dataset_nef.py  # Класс датасета `CustomNEFPairDataset`
│   ├── model_utils.py           # Создание модели NAFNet с обёрткой JDSR
│   ├── training_utils.py        # Вспомогательные функции обучения
│   ├── training_losses.py       # L1 + FFL с прогревом
│   ├── training_logger.py       # Логирование метрик
│   ├── training_validator.py    # Валидация (SSIM, PSNR)
│   ├── training_checkpoint.py   # Сохранение/загрузка чекпоинтов
│   └── ...                      # Прочие модули (визуализация, smoke‑тест, видео)
├── prepare_dataset.py           # Генерация датасета
├── run_pipeline.py              # Предобученческие проверки (визуализация, smoke‑тест)
├── run_training.py              # Запуск обучения
├── evaluate_baseline.py         # Оценка классических методов
├── plot_metrics.py              # Построение графиков
├── process_video.py             # Инференс на видео
└── docs/                        # Документация (см. ниже)
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install torch torchvision basicsr opencv-python scikit-image tifffile pillow matplotlib pandas pyyaml tqdm albumentations scipy imageio
```

### 2. Подготовка исходных изображений

Поместите резкие RGB‑изображения (можно NEF, CR2, PNG, JPG) в папку, указанную в конфиге как `source_images_dir`. По умолчанию это `source_images/`.

### 3. Настройка конфигурации

Скопируйте пример конфига (например, `configs/RAW_NAFNet_Final_100.yml`) и отредактируйте пути, параметры деградации и обучения.  
**Важное соотношение:** `network_g.upscale_factor = process_data.downscale_factor × 2`.  
**Размеры:** `gt_size = upscale_factor × lq_size`.

### 4. Генерация датасета

```bash
python prepare_dataset.py -opt configs/your_config.yml --clean-dataset
```

Флаг `--clean-dataset` полностью пересоздаёт папки `train/` и `test/`. После генерации в `dataset_root/train/lq_inputs` появятся файлы `*_bayer.tiff`, в `hq_targets` – соответствующие `*.png`.

### 5. Запуск обучения

```bash
python run_training.py -opt configs/your_config.yml
```

- Обучение автоматически продолжится с последнего чекпоинта (если включён `auto_resume`).
- Первые `ffl_start_epoch` эпох используется только L1‑лосс, затем подключается FFL.
- Логи пишутся в `experiments/{name}/train_metrics.csv` и `val_metrics.csv`.
- Валидация (SSIM и PSNR) выполняется каждую эпоху (или с заданной частотой).

### 6. Оценка качества

```bash
# Baseline (билинейная и MHC демозаика)
python evaluate_baseline.py -opt configs/your_config.yml

# Построение графиков
python plot_metrics.py -opt configs/plot_metrics.yml
```

---

## 📊 Основные метрики

| Метрика | Источник | Целевое значение (пример) |
|---------|----------|----------------------------|
| PSNR (тренировочный) | `train_metrics.csv` | > 34 дБ (на сложных датасетах) |
| PSNR (валидационный) | `val_metrics.csv` | на 1–2 дБ ниже тренировочного |
| SSIM | `val_metrics.csv` | > 0.95 |
| L1 loss | `train_metrics.csv` | монотонно убывает до 0.01–0.02 |
| FFL loss | `train_metrics.csv` | после прогрева растёт, затем снижается |
| ΔPSNR, grad_var, tv_ratio | `train_metrics.csv` | для контроля стабильности |

---

## 📚 Документация

Подробные сведения о каждом компоненте собраны в отдельных файлах (папка `docs/` или в корне):

| Файл | Содержание |
|------|-------------|
| [PIPELINE.md](PIPELINE.md) | Пошаговые инструкции по запуску всех этапов |
| [TRAINING.md](TRAINING.md) | Гиперпараметры, мониторинг, интерпретация метрик |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Структура кода, потоки данных, таблица модулей |
| [ALGORITHM.md](ALGORITHM.md) | Описание алгоритмов (деградация, демозаика, FFL) |
| [UPDOWN_SCALE.md](UPDOWN_SCALE.md) | Детальное объяснение масштабирования (downscale/upscale) |
| [CHANGES.md](CHANGES.md) | Журнал архитектурных изменений |
| [PLOT.md](PLOT.md) | Настройка визуализации метрик |

---

## 🧪 Технологический стек

- **Python** 3.9+
- **PyTorch** (CUDA 11.8+)
- **Basicsr** (архитектура NAFNet)
- **OpenCV**, **scikit‑image**, **PIL**, **tifffile**
- **YAML** (конфигурация)
- **TensorBoard** (опционально)

---

## 📝 Лицензия и авторство

Проект разработан в рамках диссертационного исследования. Для некоммерческого использования разрешается свободное копирование и модификация с указанием авторства.

---

## 🙏 Благодарности

- Авторам NAFNet [Chen et al., ECCV 2022] за предоставленный код.
- Создателям Focal Frequency Loss [Jiang et al., ICCV 2021] за идею частотной регуляризации.
- Сообществу PyTorch и Basicsr.

---

**Дата последнего обновления:** 2026-06-07  
**Версия документа:** 2.0
