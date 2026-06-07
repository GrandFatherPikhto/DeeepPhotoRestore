# ARCHITECTURE.md – Архитектура программного комплекса DeepPhotoRestore

**Версия:** 2.0 (актуальная, 2026-06-07)  
**Статус:** Полное соответствие коду после рефакторинга.

В данном документе описывается структура кода, потоки данных и взаимодействие модулей. Он предназначен для быстрого понимания того, какой файл за что отвечает, как тензоры преобразуются от RAW‑входа до RGB‑выхода и как устроен цикл обучения.

> **Навигация:** Для псевдокода алгоритмов см. [ALGORITHM.md](ALGORITHM.md). Для конфигурации – [CONFIG.md](CONFIG.md) (если есть). Для запуска обучения – `run_training.py -opt config.yml`.

---

## 1. Общая архитектура конвейера

Проект разделён на **четыре основных этапа**, каждый из которых запускается отдельным скриптом верхнего уровня:

| Скрипт | Фаза | Назначение |
|--------|------|-------------|
| `prepare_dataset.py` | Генерация датасета | Создаёт пары (LQ, HQ) из исходных резких изображений. |
| `run_pipeline.py` | Предобученческие проверки | Визуальный контроль, smoke‑тест градиентов (использует сгенерированный датасет). |
| `run_training.py` | Обучение модели | Загружает датасет, инициализирует модель, оптимизатор, лосс, запускает цикл эпох, логирует метрики, сохраняет чекпоинты, валидирует. |
| `evaluate_baseline.py` | Бенчмарк классических методов | Применяет билинейную и MHC демозаику к тестовым LQ, вычисляет PSNR/SSIM. |
| `plot_metrics.py` | Построение графиков | Визуализирует логи обучения (loss, PSNR, SSIM, LR и др.). |
| `process_video.py` | Инференс на видео | Покадрово восстанавливает видео, применяя ту же модель деградации (опционально) и обученную сеть. |

**Библиотеки (`libraries/`)** содержат переиспользуемые модули, которые вызываются из этих скриптов.

---

## 2. Ключевые модули `libraries/` и их ответственность

| Файл | Отвечает за | Используется в |
|------|-------------|----------------|
| `config.py` | Парсинг аргументов командной строки, загрузка YAML‑конфига, подстановка `{name}` в пути, проверка обязательности `network_g.upscale_factor`, предупреждение об устаревшем `datasets.train.upscale_factor`. | Все скрипты |
| `logger.py` | Настройка двухпоточного логгирования (консоль + файл). | Все скрипты |
| `device.py` | Определение устройства (CUDA/CPU). | `run_pipeline.py`, `run_training.py` |
| `pipeline_prepare.py` | Проверка/очистка датасета, вызов генерации (`process_source_images`). | `run_pipeline.py` |
| `pipeline_generation_io.py` | Чтение исходных изображений, вызов `generate_lq_from_hq` для каждого файла, сохранение LQ/HQ в `lq_inputs/`, `hq_targets/`. | `pipeline_prepare.py` |
| `pipeline_generation_core.py` | **Ядро деградации:** оптическое уменьшение (`downscale_factor`), PSF‑размытие, маска Байера, добавление шума, упаковка в 4 канала. | `pipeline_generation_io.py`, `video_degradation.py` |
| `degradation_ops.py` | Низкоуровневые операции: создание PSF‑ядра, свёртка, добавление шума (коррелированного/некоррелированного), маска Байера, извлечение субканалов. | `pipeline_generation_core.py` |
| `training_dataset_nef.py` | Класс `CustomNEFPairDataset` – загрузка пар LQ/HQ, случайный кроп, синхронные аугментации (flip, rot90 отключены). Читает `upscale_factor` из `network_g`. | `run_training.py` (через `training_utils.create_train_loader`) |
| `model_utils.py` | Создание NAFNet с обёрткой `NAFNetDemosaicSuperResolutionWrapper`, загрузка предобученных весов (частичный перенос с `strict=False`). `upscale_factor` берётся из `network_g`. | `run_pipeline.py`, `run_training.py` |
| `training_utils.py` | Вспомогательные функции: `create_train_loader`, `create_optimizer_and_scheduler`, `compute_metrics` (с передачей `current_epoch`), `log_progress`, `cleanup_experiment`. | `run_training.py` |
| `training_losses.py` | Комбинированная потеря `CombinedLoss` (L1 + FFL). Поддерживает `ffl_start_epoch` (прогрев: до указанной эпохи FFL отключён). | `run_training.py` |
| `training_logger.py` | Класс `TrainingLogger` – логирование в CSV (`train_metrics.csv`, `val_metrics.csv`), текстовый файл, TensorBoard. | `run_training.py` |
| `training_validator.py` | Класс `VisualValidator` – валидация на тестовой выборке (центральный кроп, инференс, расчёт SSIM **и PSNR**), сохранение предсказаний. Использует `upscale_factor` из `network_g`. | `run_training.py` |
| `training_checkpoint.py` | Сохранение и загрузка чекпоинтов с поддержкой аварийного возобновления (флаг `is_emergency`). | `run_training.py` |
| `pipeline_visuals.py` | Визуальный контроль: билинейная демозаика LQ (превью), сохранение LQ и HQ изображений с коррекцией цветового пространства (RGB→BGR). | `run_pipeline.py` |
| `pipeline_smoke.py` | Smoke‑тест: один прямой и обратный проход с комбинированным лоссом для проверки градиентов и VRAM. | `run_pipeline.py` |
| `evaluate_baseline.py` | Реализация билинейной и MHC демозаики, расчёт PSNR/SSIM. | `evaluate_baseline.py` (скрипт верхнего уровня) |
| `video_utils.py` | Открытие/запись видео, конвертация кадров в тензоры и обратно. | `process_video.py` |
| `video_degradation.py` | Применение `generate_lq_from_hq` к кадру видео и ресайз до целевого размера. | `process_video.py` |
| `video_inference.py` | Загрузка модели, цикл покадровой обработки видео. | `process_video.py` |

---

## 3. Поток тензоров: от RAW‑файла до предсказания сети

### 3.1. Генерация синтетического LQ (физическая деградация)

Вход: **резкое RGB** – `hq_rgb` (uint8, H×W×3) из папки `source_images_dir`.

**Шаги (функция `generate_lq_from_hq` в `pipeline_generation_core.py`):**

1. **Вырезание центрального HQ-патча** размера `gt_size` (из конфига). Координаты выравниваются по модулю 4 для сохранения фазы RGGB.
2. **Оптическое уменьшение** в `downscale_factor` раз (билинейная интерполяция).  
   `hq_small = cv2.resize(hq_patch, (gt_size//downscale, gt_size//downscale))` → float32 [0,1].
3. **PSF‑размытие** (свёртка с гауссианой, sigma = `psf_sigma`):  
   `hq_blurred = conv2d(hq_small, psf_kernel)` → float32 [0,1].
4. **Маска Байера (RGGB)**:  
   `bayer = apply_bayer_mask_float(hq_blurred)` → массив (gt_size//downscale, gt_size//downscale) float32.
5. **Добавление шума** (если `add_noise = true`):  
   - Для некоррелированного шума: `bayer += N(0, sqrt(noise_power))`.  
   - Для коррелированного: сначала белый шум, затем свёртка с PSF-ядром.
6. **Упаковка в 4 канала** (субдискретизация):  
   `lq_packed = extract_bayer_subchannels(bayer)` → (gt_size//downscale//2, gt_size//downscale//2, 4) float32.
7. **Сохранение** как uint16 TIFF в `train/lq_inputs/` или `test/lq_inputs/`.

**Итоговое сжатие:** общее уменьшение линейного размера от HQ до LQ = `downscale_factor × 2`.  
Например, `downscale_factor=2` → LQ в 4 раза меньше HQ.

### 3.2. Загрузка данных в обучении (класс `CustomNEFPairDataset`)

- **LQ**: читается TIFF → float32 / 65535 → (h_lq, w_lq, 4).
- **HQ**: читается PNG → float32 / 255 → (h_hq, w_hq, 3).

**Проверка геометрии:**  
`assert h_hq == upscale_factor * h_lq and w_hq == upscale_factor * w_lq`.  
`upscale_factor` берётся из `opt['network_g']['upscale_factor']` (единый источник).

**Кроп:**  
- Случайные координаты `(top_lq, left_lq)` для патча `lq_size × lq_size`.
- Соответствующий кроп в HQ: `top_hq = top_lq * upscale_factor`, `left_hq = left_lq * upscale_factor`, размер `gt_size = upscale_factor * lq_size`.

**Аугментации (синхронные):**  
- `torchvision.transforms.functional.hflip / vflip` (вероятность 0.5).  
- Повороты на 90° отключены (так как нарушают порядок каналов RGGB).

**Выход:**  
- LQ тензор: `(4, lq_size, lq_size)` float32 [0,1].
- HQ тензор: `(3, gt_size, gt_size)` float32 [0,1].

### 3.3. Прямой проход модели (`NAFNetDemosaicSuperResolutionWrapper`)

Модель состоит из:
1. **Базовый NAFNet** (вход 4 канала, выход 4 канала, без изменения пространственного размера).
2. **Upsample block**: `Conv2d(4, out_ch * upscale_factor², 3,1) + PixelShuffle(upscale_factor)`.
3. **Адаптер каналов** для совместимости с предобученными весами SIDD (4→3 и обратно).

**Поток:**  
`lq` (4, lq_size, lq_size) → NAFNet → `mid` (4, lq_size, lq_size) → Upsample block → `out` (3, lq_size*upscale, lq_size*upscale).

Для `upscale_factor=4` и `lq_size=128` выход будет `(3, 512, 512)`, что соответствует `gt_size`.

**Выход сети:** `pred` (B, 3, gt_size, gt_size) в диапазоне `[0,1]` (без сигмоиды, loss сам ограничивает).

### 3.4. Расчёт лосса и обратное распространение

Используется `CombinedLoss` из `training_losses.py`:

- **L1Loss** между `pred` и `target` (MAE).
- **FocalFrequencyLoss** (логарифмическая версия по умолчанию, `ffl_type="log"`):
  - БПФ → амплитуды → логарифмическое сжатие → взвешенная квадратичная ошибка с фокальным весом.
- **Прогрев:** параметр `ffl_start_epoch` (в конфиге, например 20). Пока `current_epoch < ffl_start_epoch`, `ffl_weight` эффективно равен 0.
- `total_loss = l1_weight * l1 + ffl_weight * ffl`.

Затем:
1. `total_loss.backward()`
2. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` – стабилизация градиентов.
3. `optimizer.step()`

---

## 4. Процесс обучения (пошагово)

Цикл обучения реализован в `run_training.py`. Ниже – последовательность действий в рамках одной эпохи:

```
for epoch in range(start_epoch, num_epochs):
    model.train()
    for batch_idx, (lq, hq) in enumerate(train_loader):
        lq, hq = lq.to(device), hq.to(device)
        optimizer.zero_grad()
        
        pred = model(lq)                     # forward
        
        metrics = compute_metrics(pred, hq, criterion, loss_type, current_epoch=epoch)
        total_loss = metrics['total_loss']
        
        total_loss.backward()                # backward
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # логирование (CSV, текст, TensorBoard)
        train_logger.log_metrics(...)
        log_progress(...)
        
        global_step += 1
    
    scheduler.step()                         # обновление LR
    
    if (epoch+1) % validation_freq == 0:
        mean_ssim, mean_psnr = validator.run_validation(model, epoch, device)
        train_logger.log_validation_metrics(epoch, mean_ssim, mean_psnr)
    
    if (epoch+1) % save_checkpoint_epoch == 0:
        save_checkpoint(...)
```

**Где что находится:**
- `train_loader` создаётся через `training_utils.create_train_loader`, который вызывает `CustomNEFPairDataset`.
- `model` создаётся через `model_utils.create_nafnet_model` (с возможностью частичной загрузки предобученных весов).
- `criterion` – `CombinedLoss` (если `losses.type = "combined"`) или `L1Loss`.
- `optimizer` – AdamW, `scheduler` – MultiStepLR (или другой).
- `train_logger` – экземпляр `TrainingLogger` (пишет в CSV, текст, TensorBoard).
- `validator` – `VisualValidator` (валидация с центральным кропом, возвращает SSIM и PSNR).

---

## 5. Валидация (центральный кроп, SSIM и PSNR)

1. Для каждого файла в `test/lq_inputs/` загружается 4‑канальный TIFF.
2. Вырезается **центральный кроп** размера `lq_size` (координаты `(H/2 - lq_size/2, W/2 - lq_size/2)`).
3. Инференс модели → получаем RGB размером `gt_size = upscale_factor * lq_size`.
4. Из соответствующего HQ (PNG) вырезается центральный кроп с координатами, умноженными на `upscale_factor`, и размером `gt_size`.
5. Сравнение:
   - **SSIM** через `skimage.metrics.structural_similarity` (data_range=255).
   - **PSNR** через собственный расчёт MSE и формулу: `20 * log10(255 / sqrt(MSE))`.
6. Результаты сохраняются в `val_metrics.csv` (колонки `epoch`, `ssim`, `psnr`), средние значения выводятся в лог.

---

## 6. Связь между файлами и алгоритмом (таблица)

| Алгоритмический шаг | Где реализован (файл → функция/класс) |
|---------------------|----------------------------------------|
| Генерация LQ из HQ | `pipeline_generation_core.py` → `generate_lq_from_hq` |
| PSF‑ядро, свёртка, шум | `degradation_ops.py` → `create_psf_kernel`, `add_correlated_noise_float`, `apply_bayer_mask_float` |
| Упаковка Байера в 4 канала | `degradation_ops.py` → `extract_bayer_subchannels` |
| Загрузка пар LQ/HQ с кропом и аугментациями | `training_dataset_nef.py` → `CustomNEFPairDataset` |
| Архитектура NAFNet + PixelShuffle | `model_utils.py` → `NAFNetDemosaicSuperResolutionWrapper`, `create_nafnet_model` |
| Комбинированная потеря (L1+FFL) с прогревом | `training_losses.py` → `CombinedLoss`, `FocalFrequencyLoss`, `FocalFrequencyLossLog` |
| Цикл обучения | `run_training.py` → `main` |
| Логирование метрик (CSV, TensorBoard) | `training_logger.py` → `TrainingLogger` |
| Валидация (SSIM+PSNR) | `training_validator.py` → `VisualValidator.run_validation` |
| Чекпоинты (сохранение/загрузка с emergency) | `training_checkpoint.py` → `save_checkpoint`, `load_checkpoint` |
| Билинейная и MHC демозаика (baseline) | `evaluate_baseline.py` → `bilinear_demosaic`, `mhc_demosaic` |
| Визуальный контроль (превью) | `pipeline_visuals.py` → `run_visual_control`, `bilinear_demosaic_rggb` |
| Smoke‑тест | `pipeline_smoke.py` → `run_smoke_test` |
| Инференс на видео | `video_inference.py` → `process_video`, `video_degradation.py` → `degrade_frame` |

---

## 7. Рекомендации по навигации в коде

- **Хотите изменить физику деградации** (PSF, шум, масштабирование)? → `pipeline_generation_core.py` и `degradation_ops.py`.
- **Хотите добавить новую аугментацию**? → `training_dataset_nef.py` (метод `__getitem__`).
- **Хотите модифицировать архитектуру сети** (ширина, глубина)? → измените секцию `network_g` в YAML; сама обёртка – `model_utils.py`.
- **Хотите изменить функцию потерь** (веса, эпоху прогрева)? → `training_losses.py` и секция `losses` в конфиге.
- **Хотите добавить метрику в логи**? → `training_logger.py` и секция `logger` в конфиге.
- **Хотите изменить частоту валидации или сохранения**? → параметры `validation_freq`, `save_checkpoint_epoch` в конфиге.

---

## 8. Типичный поток данных (с примерами размерностей)

Возьмём конфиг: `downscale_factor=2`, `upscale_factor=4`, `gt_size=512`, `lq_size=128`.

| Этап | Размерность | Тип | Примечание |
|------|-------------|-----|-------------|
| Исходное RGB (source) | (H, W, 3) | uint8 | H, W – любые, например 4000×6000 |
| Центральный HQ-патч | (512, 512, 3) | uint8 | вырезан и сохранён как PNG |
| После downscale (2) | (256, 256, 3) | float32 | `512 // 2 = 256` |
| После маски Байера | (256, 256) | float32 | одноканальный Bayer |
| После упаковки (LQ на диске) | (128, 128, 4) | uint16 | `256 // 2 = 128` |
| Кроп LQ в датасете | (128, 128, 4) | float32 [0,1] | `lq_size=128` |
| Вход в NAFNet | (4, 128, 128) | float32 | тензор после transpose |
| Выход NAFNet (латентный) | (4, 128, 128) | float32 | без изменения пространства |
| После Conv2d+PixelShuffle(4) | (3, 512, 512) | float32 | `128*4 = 512` |
| Таргет (HQ кроп) | (3, 512, 512) | float32 | загружен из PNG |
| Лосс (L1 + FFL) | скаляр | float | – |

---

## 9. Заключение

Данная архитектура обеспечивает:
- **Модульность** – каждый этап (генерация, загрузка, модель, лосс, валидация, логирование) выделен в отдельный файл/класс.
- **Конфигурируемость** – поведение меняется через YAML без правки кода.
- **Физическую корректность** – деградация соответствует реальному тракту (оптическое уменьшение, PSF, Байер, коррелированный шум).
- **Стабильность обучения** – клиппинг градиентов, прогрев FFL, аварийное возобновление.
- **Полноту метрик** – логируются train loss, PSNR, LR, градиенты, TV ratio, VRAM; validation – SSIM и PSNR.

Для детального изучения каждого компонента используйте ссылки на соответствующие файлы в таблице выше.

---

**Дата последнего обновления:** 2026-06-07  
**Версия:** 2.0 (полное соответствие коду)
```

Этот документ теперь отражает все исправления, включая:
- Единый `upscale_factor` из `network_g`.
- Использование `downscale_factor` в генерации.
- Прогрев FFL (`ffl_start_epoch`).
- Валидатор, возвращающий PSNR.
- Клиппинг градиентов.
- Удаление устаревшего датасета.
- Правильные пути в `run_pipeline.py`.
- Импорт `math` и т.д.
