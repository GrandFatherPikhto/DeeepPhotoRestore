# CONFIG.md – Полный справочник по конфигурационному YAML-файлу

Конфигурационный файл (например, `options/train/RAW_NAFNet_Test_20.yml`) используется **одновременно** скриптами `run_pipeline.py` (генерация датасета, визуальный контроль, smoke‑тест) и `run_training.py` (обучение). Ниже приведены все секции с пояснениями, какие параметры к какому скрипту относятся.

> **Примечание:** Пути, содержащие `{name}`, автоматически заменяются на значение из поля `name`.

---

## 1. Общие настройки (используются обоими скриптами)

| Параметр | Тип | Описание | Пример |
|----------|-----|----------|--------|
| `name` | строка | Имя эксперимента. Подставляется в пути, содержащие `{name}`. | `RAW-NAFNet-Test-20` |
| `manual_seed` | целое | Фиксация генератора случайных чисел (воспроизводимость). | `42` |

---

## 2. Секция `path` (используется и pipeline, и training)

| Параметр | Скрипт | Описание | Пример |
|----------|--------|----------|--------|
| `source_images_dir` | pipeline | Папка с исходными резкими изображениями для генерации датасета. | `"/home/grand/Images/NEF_Test_20"` |
| `dataset_root` | pipeline, training | Корень сгенерированного датасета (внутри создаются `train/`, `test/`). | `"datasets/nef_nafnet_20"` |
| `resume_path` | training | Путь для сохранения / восстановления чекпоинта (поддерживает `{name}`). | `"experiments/{name}/checkpoints/resume_nef_nafnet_20.pth"` |
| `pretrain_network_g` | training | (опционально) Путь к предобученным весам NAFNet (например, SIDD). | `"pretrained/NAFNet-SIDD-width32.pth"` |

---

## 3. Секция `process_data` – **только для `run_pipeline.py`** (генерация датасета)

Определяет физическую деградацию, применяемую к исходным HQ для получения LQ.

| Параметр | Тип | Описание | Пример |
|----------|-----|----------|--------|
| `downscale_factor` | целое | Во сколько раз уменьшить разрешение перед формированием LQ. Для полного восстановления с `upscale_factor` в сети должно быть `upscale_factor = 2 * downscale_factor` (см. п. 12 ТЗ). | `4` |
| `noise.add` | логическое | Добавлять ли шум. | `true` |
| `noise.snr_db` | число | Отношение сигнал/шум в децибелах. | `20` |
| `noise.correlated` | логическое | Использовать коррелированный шум (свёртка с PSF). | `true` |
| `noise.psf_sigma` | число | Сигма гауссианы для PSF (размытие). | `0.5` |

---

## 4. Секция `network_g` – архитектура NAFNet (используется и pipeline, и training)

| Параметр | Описание | Пример |
|----------|----------|--------|
| `type` | Тип сети (всегда `NAFNet`). | `NAFNet` |
| `arch_file` | Путь к файлу архитектуры (из субмодуля NAFNet). | `"modules/NAFNet/basicsr/models/archs/NAFNet_arch.py"` |
| `class_name` | Имя класса в файле архитектуры. | `"NAFNet"` |
| `num_in_ch` | Число входных каналов (для RGGB = 4). | `4` |
| `num_out_ch` | Число выходных каналов (RGB = 3). | `3` |
| `width` | Базовая ширина каналов (количество фильтров в первом слое). | `32` |
| `enc_blk_nums` | Список: количество блоков NAFNet на этапах энкодера. | `[2, 2, 4, 8]` |
| `middle_blk_num` | Количество блоков в латентном пространстве (bottleneck). | `12` |
| `dec_blk_nums` | Список: количество блоков NAFNet на этапах декодера. | `[2, 2, 2, 2]` |
| `upscale_factor` | Коэффициент увеличения разрешения (PixelShuffle). Должен быть согласован с `downscale_factor` и размерами патчей. | `2` (стандарт) или `8` (для `downscale_factor=4` по п. 12 ТЗ) |

> **Важно:** Соотношение размеров патчей в датасете: `gt_size = upscale_factor * lq_size`. Для `upscale_factor=2` классический случай, для `upscale_factor=8` – при `downscale_factor=4`.

---

## 5. Секция `datasets` – **только для `run_training.py`** (загрузка данных)

| Параметр | Описание | Пример |
|----------|----------|--------|
| `train.type` | Имя класса датасета (должен быть `CustomNEFPairDataset`). | `CustomNEFPairDataset` |
| `train.batch_size_per_gpu` | Размер батча на один GPU. | `16` |
| `train.num_worker_per_gpu` | Количество процессов загрузки данных. | `4` |
| `train.auto_resume` | Автоматически продолжать обучение из чекпоинта (если найден). | `true` |
| `train.gt_size` | Размер выходного RGB‑патча. **Должен быть равен `upscale_factor * lq_size`**. | `256` (при `upscale_factor=2`) или `1024` (при `upscale_factor=8`) |
| `train.lq_size` | Размер входного упакованного RGGB‑патча. | `128` |
| `train.use_flip` | Включить случайные отражения (горизонтальные/вертикальные). | `true` |
| `train.use_rot` | Включить случайные повороты на 90/180/270 градусов. | `true` |

> **Примечание:** Параметр `save_checkpoint_epoch` в секции `datasets` устарел; используйте одноимённый в секции `train`.

---

## 6. Секция `train` – **только для `run_training.py`** (гиперпараметры обучения)

| Параметр | Описание | Пример |
|----------|----------|--------|
| `num_epochs` | Общее количество эпох. | `200` |
| `validation_freq` | Как часто (эпох) выполнять валидацию. | `1` |
| `save_checkpoint_epoch` | Как часто (эпох) сохранять чекпоинт. | `5` |
| `optim_g.lr` | Начальная скорость обучения. | `1e-3` |
| `optim_g.weight_decay` | L2‑регуляризация (weight decay). | `1e-4` |
| `optim_g.betas` | Параметры AdamW `(beta1, beta2)`. | `[0.9, 0.999]` |
| `scheduler.eta_min` | Минимальная скорость обучения в конце (CosineAnnealingLR). | `1e-7` |

---

## 7. Секция `losses` – **только для `run_training.py`** (функция потерь)

| Параметр | Описание | Пример |
|----------|----------|--------|
| `type` | Тип лосса: `"combined"` (L1 + FFL) или `"l1"`. | `combined` |
| `l1_weight` | Вес L1 компоненты. | `1.0` |
| `ffl_weight` | Вес Focal Frequency Loss. | `0.5` |
| `ffl_alpha` | Параметр фокусировки в FFL (≥0). | `1.0` |
| `ffl_log_factor` | Фактор логарифмического сжатия спектра (при `ffl_type="log"`). | `100.0` |
| `ffl_type` | Тип FFL: `"linear"` (стандартный) или `"log"` (логарифмический, стабильнее). | `"log"` |

---

## 8. Секция `logger` – **только для `run_training.py`** (логирование метрик)

| Параметр | Описание | Пример |
|----------|----------|--------|
| `print_freq` | Частота вывода в консоль (каждые N шагов). | `2` |
| `use_tb_logger` | Включить TensorBoard. | `true` |
| `log_file_name` | Имя текстового файла с метриками (в папке эксперимента). | `"train_progress.log"` |
| `csv_file_name` | Имя CSV‑файла с метриками. | `"train_metrics.csv"` |
| `log_csv` | Записывать ли в CSV. | `true` |
| `file_log_freq` | Частота записи в текстовый лог (шаги). | `1` |
| `log_loss_components` | Записывать отдельно L1 и FFL. | `true` |
| `include_in_console` | Какие метрики выводить в консоль (словарь). | `loss: true, psnr: true, vram: true` |
| `include_in_file_log` | Какие метрики сохранять в текстовый файл (словарь). | `loss: true, psnr: true, log_ssim: true` |

---

## 9. Секция `visuals_logger` – используется `run_pipeline.py` (визуальный контроль)

| Параметр | Описание | Пример |
|----------|----------|--------|
| `output_dir` | Папка для сохранения превью LQ/HQ. Поддерживает `{name}`. | `"experiments/{name}/debug_visuals"` |
| `lq_preview_name` | Имя файла для LQ‑превью (после билинейной демозаики). | `"debug_lq_preview.png"` |
| `gt_reference_name` | Имя файла для HQ‑эталона. | `"debug_gt_reference.png"` |

---

## 10. Секция `pipeline_logger` – используется обоими скриптами (общий лог)

| Параметр | Описание | Пример |
|----------|----------|--------|
| `log_file` | Путь к файлу для сообщений (предупреждения, ошибки, SSIM). | `"experiments/{name}/pipeline.log"` |

---

## Пример полного конфигурационного файла (актуальный)

```yaml
name: RAW-NAFNet-Test-20
manual_seed: 42

path:
  source_images_dir: "/home/grand/Images/NEF_Test_20"
  dataset_root: "datasets/nef_nafnet_20"
  resume_path: "experiments/{name}/checkpoints/resume_nef_nafnet_20.pth"
  # pretrain_network_g: "pretrained/NAFNet-SIDD-width32.pth"

process_data:
  downscale_factor: 4
  noise:
    add: true
    snr_db: 20
    correlated: true
    psf_sigma: 0.5

network_g:
  type: NAFNet
  arch_file: "modules/NAFNet/basicsr/models/archs/NAFNet_arch.py"
  class_name: "NAFNet"
  num_in_ch: 4
  num_out_ch: 3
  width: 32
  enc_blk_nums: [2, 2, 4, 8]
  middle_blk_num: 12
  dec_blk_nums: [2, 2, 2, 2]
  upscale_factor: 2   # для downscale_factor=4 должно быть 8, но в данном конфиге 2 (пример)

datasets:
  train:
    type: CustomNEFPairDataset
    batch_size_per_gpu: 16
    num_worker_per_gpu: 4
    auto_resume: true
    gt_size: 256
    lq_size: 128

train:
  validation_freq: 1
  save_checkpoint_epoch: 5
  num_epochs: 200
  optim_g:
    lr: 1e-3
    weight_decay: 1e-4
    betas: [0.9, 0.999]
  scheduler:
    eta_min: 1e-7

losses:
  type: "combined"
  l1_weight: 1.0
  ffl_weight: 0.5
  ffl_alpha: 1.0
  ffl_log_factor: 100.0
  ffl_type: "log"

logger:
  print_freq: 2
  use_tb_logger: true
  log_file_name: "train_progress.log"
  csv_file_name: "train_metrics.csv"
  log_csv: true
  file_log_freq: 1
  log_loss_components: true
  include_in_console:
    loss: true
    psnr: true
    vram: true
    learning_rate: false
  include_in_file_log:
    loss: true
    psnr: true
    learning_rate: true
    delta_psnr: true
    grad_variance: true
    tv_ratio: true
    vram_usage: true
    log_ssim: true

visuals_logger:
  output_dir: "experiments/{name}/debug_visuals"
  lq_preview_name: "debug_lq_preview.png"
  gt_reference_name: "debug_gt_reference.png"

pipeline_logger:
  log_file: "experiments/{name}/pipeline.log"
```

---

## Важные замечания

- **Соотношение размеров патчей:** `gt_size` **обязан** быть равен `upscale_factor * lq_size`. Это требование модели (PixelShuffle увеличивает разрешение в `upscale_factor` раз).
- **Согласованность деградации и апскейла:** Если `process_data.downscale_factor = 4`, то для полного восстановления исходного масштаба рекомендуется `network_g.upscale_factor = 8` (сжатие в 4 раза + упаковка Байера даёт суммарное уменьшение в 8 раз). В приведённом примере `upscale_factor = 2` – это классический случай демозаики без дополнительного сжатия.
- **Воспроизводимость:** Фиксированный `manual_seed` гарантирует одинаковое разбиение train/test и аугментации.
- **Предобученные веса:** Строка `pretrain_network_g` закомментирована – при необходимости раскомментируйте и укажите путь к `.pth` файлу (например, `NAFNet-SIDD-width32.pth`).

---

**Дата последнего обновления:** 2026-06-06  
**Версия:** 2.0 (добавлен `upscale_factor`, исправлено соотношение размеров, актуализирован пример)
