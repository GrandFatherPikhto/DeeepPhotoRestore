# CONFIG.md – Конфигурация эксперимента

Все настройки конвейера (генерация датасета, архитектура сети, обучение, логирование) задаются в YAML-файле, путь к которому передаётся скриптам через аргумент `-opt`. Ниже приведено описание всех секций и параметров.

---

## Секция `name`

Обязательный параметр. Имя эксперимента. Используется для создания подпапки в `experiments/` и для подстановки `{name}` в пути (см. раздел о подстановке).

```yaml
name: NAFNet-100-02
```

---

## Секция `manual_seed`

Фиксация генератора случайных чисел для воспроизводимости.

```yaml
manual_seed: 42
```

---

## Секция `path`

Пути к данным и чекпоинтам.

| Параметр | Описание | Пример |
|----------|----------|--------|
| `source_images_dir` | Папка с исходными резкими RGB‑изображениями (NEF, PNG, JPEG и др.) | `"/home/user/Images/NEF_100"` |
| `dataset_root` | Корневая папка, где будут созданы `train/` и `test/` с подпапками `lq_inputs`/`hq_targets` | `"datasets/{name}"` |
| `resume_path` | Путь для сохранения/загрузки чекпоинта. Может содержать `{name}`. | `"experiments/{name}/checkpoints/resume.pth"` |
| `pretrain_network_g` (опционально) | Путь к предобученной модели NAFNet (например, SIDD). | `"pretrained/NAFNet-SIDD-width32.pth"` |

```yaml
path:
  source_images_dir: "/home/grand/Images/NEF_Test_100"
  dataset_root: "datasets/{name}"
  resume_path: "experiments/{name}/checkpoints/resume.pth"
  pretrain_network_g: "pretrained/NAFNet-SIDD-width32.pth"   # опционально
```

---

## Секция `datasets`

Параметры датасета и загрузчика.

### `datasets.train`

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `gt_size` | Размер квадратного HQ‑патча (эталон, в пикселях) | 256 |
| `lq_size` | Размер квадратного LQ‑патча (упакованный RGGB, 4 канала). Должен удовлетворять `gt_size = upscale_factor * lq_size`. | 128 |
| `batch_size_per_gpu` | Размер батча на одну GPU | 2 |
| `num_worker_per_gpu` | Количество процессов для загрузки данных | 4 |
| `use_flip` | Включить горизонтальные/вертикальные отражения (безопасно для Bayer) | false |
| `use_rot` | Включить повороты на 90° (обычно **false**, т.к. нарушают RGGB) | false |
| `auto_resume` | Автоматически продолжать обучение с последнего чекпоинта | true |

**Примечание:** `upscale_factor` в этой секции **устарел** и игнорируется. Используйте единый `network_g.upscale_factor`.

```yaml
datasets:
  train:
    gt_size: 512
    lq_size: 128
    batch_size_per_gpu: 2
    num_worker_per_gpu: 2
    use_flip: true
    use_rot: false
    auto_resume: true
```

---

## Секция `process_data`

Моделирование физической деградации при генерации LQ из HQ.

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `downscale_factor` | Оптическое уменьшение разрешения перед наложением маски Байера. Общее сжатие = `downscale_factor × 2`. | 2 |
| `noise.add` | Добавлять шум? | false |
| `noise.snr_db` | Отношение сигнал/шум в децибелах | 30 |
| `noise.correlated` | Коррелированный шум (свёртка с PSF) | false |
| `noise.psf_sigma` | Сигма гауссова ядра PSF для размытия и коррелированного шума | 1.5 |

```yaml
process_data:
  downscale_factor: 2
  noise:
    add: true
    snr_db: 20
    correlated: true
    psf_sigma: 0.5
```

---

## Секция `network_g`

Архитектура модели NAFNet и обёртки JDSR.

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `num_in_ch` | Число входных каналов (для RGGB всегда 4) | 4 |
| `num_out_ch` | Число выходных каналов (RGB) | 3 |
| `width` | Базовая ширина каналов NAFNet | 32 |
| `upscale_factor` | **Единый источник истины** – коэффициент масштабирования (должен быть равен `downscale_factor × 2`). | 4 |
| `middle_blk_num` | Количество блоков в центральной части | 12 |
| `enc_blk_nums` | Количество блоков на уровнях энкодера | [2,2,4,8] |
| `dec_blk_nums` | Количество блоков на уровнях декодера | [2,2,2,2] |

```yaml
network_g:
  num_in_ch: 4
  num_out_ch: 3
  width: 64
  upscale_factor: 4
  middle_blk_num: 12
  enc_blk_nums: [2, 2, 4, 8]
  dec_blk_nums: [2, 2, 2, 2]
```

---

## Секция `train`

Гиперпараметры оптимизации и расписания.

| Параметр | Описание | Пример |
|----------|----------|--------|
| `validation_freq` | Частота валидации (эпохи) | 1 |
| `save_checkpoint_epoch` | Частота сохранения чекпоинтов (эпохи) | 10 |
| `num_epochs` | Общее число эпох | 300 |
| `optim_g.type` | Тип оптимизатора (AdamW) | `AdamW` |
| `optim_g.lr` | Начальная скорость обучения | 1e-4 |
| `optim_g.weight_decay` | L2‑регуляризация | 1e-4 |
| `optim_g.betas` | Параметры Adam | [0.9, 0.999] |
| `scheduler.type` | Тип планировщика | `MultiStepLR` |
| `scheduler.milestones` | Эпохи снижения LR | [150, 225] |
| `scheduler.gamma` | Коэффициент уменьшения LR | 0.1 |

```yaml
train:
  validation_freq: 1
  save_checkpoint_epoch: 10
  num_epochs: 300
  optim_g:
    type: AdamW
    lr: 1e-4
    weight_decay: 1e-4
    betas: [0.9, 0.999]
  scheduler:
    type: MultiStepLR
    milestones: [150, 225]
    gamma: 0.1
```

---

## Секция `losses`

Функция потерь.

| Параметр | Описание | По умолчанию |
|----------|----------|-------------|
| `type` | Тип: `"l1"` или `"combined"` | `l1` |
| `l1_weight` | Вес L1‑компоненты | 1.0 |
| `ffl_weight` | Вес FFL‑компоненты | 0.2 |
| `ffl_start_epoch` | С какой эпохи включать FFL (прогрев) | 0 |
| `ffl_type` | `"linear"` или `"log"` | `log` |
| `ffl_alpha` | Фокальный параметр | 1.0 |
| `ffl_log_factor` | Коэффициент сжатия для логарифмической версии | 100.0 |

```yaml
losses:
  type: combined
  l1_weight: 1.5
  ffl_weight: 0.2
  ffl_start_epoch: 20
  ffl_alpha: 1.5
  ffl_log_factor: 100.0
  ffl_type: log
```

---

## Секция `logger`

Настройки логирования.

| Параметр | Описание | По умолчанию |
|----------|----------|-------------|
| `use_tb_logger` | Включить TensorBoard | false |
| `log_csv` | Записывать `train_metrics.csv` | false |
| `csv_file_name` | Имя CSV‑файла для обучения | `train_metrics.csv` |
| `log_file_name` | Имя текстового лог‑файла | `train_progress.log` |
| `file_log_freq` | Частота записи в текстовый лог (шаги) | 20 |
| `include_in_console` | Какие поля выводить на экран | см. пример |
| `include_in_file_log` | Какие поля записывать в `train_progress.log` | см. пример |

```yaml
logger:
  use_tb_logger: true
  log_csv: true
  csv_file_name: "train_metrics.csv"
  log_file_name: "train_progress.log"
  file_log_freq: 50
  include_in_console:
    loss: true
    psnr: true
    vram: true
    learning_rate: true
  include_in_file_log:
    loss: true
    psnr: true
    learning_rate: true
    delta_psnr: true
    grad_variance: true
    tv_ratio: true
    vram_usage: true
    log_ssim: true
```

---

## Секция `visuals_logger`

Настройки визуального контроля.

| Параметр | Описание | По умолчанию |
|----------|----------|-------------|
| `output_dir` | Папка для сохранения превью | `"experiments/{name}/debug_visuals"` |
| `lq_preview_name` | Имя файла для LQ‑превью | `"debug_lq_preview.png"` |
| `gt_reference_name` | Имя файла для HQ‑превью | `"debug_gt_reference.png"` |

```yaml
visuals_logger:
  output_dir: "experiments/{name}/debug_visuals"
  lq_preview_name: "debug_lq_preview.png"
  gt_reference_name: "debug_gt_reference.png"
```

---

## Секция `pipeline_logger`

Общий лог-файл конвейера.

| Параметр | Описание | По умолчанию |
|----------|----------|-------------|
| `log_file` | Путь к текстовому логу | `"experiments/{name}/pipeline.log"` |

```yaml
pipeline_logger:
  log_file: "experiments/{name}/pipeline.log"
```

---

## Секции для `plot_metrics.py` (опциональны)

Эти секции используются только скриптом построения графиков и не влияют на обучение.

| Параметр | Описание |
|----------|----------|
| `experiment_dir` | Папка эксперимента (где лежат `train_metrics.csv`, `val_metrics.csv`) |
| `output_dir` | Папка для сохранения графиков |
| `data_files` | Словарь с путями к CSV-файлам (по ключам) |
| `plots` | Список графиков (тип, источник, колонки, сглаживание и т.д.) |

```yaml
experiment_dir: "experiments/{name}"
output_dir: "experiments/{name}/figures"
data_files:
  train_metrics: "experiments/{name}/train_metrics.csv"
  val_metrics: "experiments/{name}/val_metrics.csv"
```

---

## Подстановка `{name}`

Во всех путях, перечисленных выше (кроме `source_images_dir`, если не указано иное), можно использовать шаблон `{name}`. При загрузке конфига он будет автоматически заменён на значение `name`. Это позволяет создавать переносимые конфиги и не дублировать имя эксперимента.

**Поддерживается в:**
- `path.dataset_root`
- `path.resume_path`
- `visuals_logger.output_dir`
- `pipeline_logger.log_file`
- `experiment_dir`
- `output_dir`
- `data_files.*`
- `plot_metrics.py` (через аргумент `--name` или поле `name` в YAML)

---

## Дополнительные примечания

1. **Обязательные параметры**  
   - `network_g.upscale_factor` – проверяется в `config.py`. При отсутствии выбрасывается исключение.
   - `name` – используется для подстановки, но может быть пустым (тогда подстановка не выполняется).

2. **Устаревшие параметры (игнорируются)**  
   - `datasets.train.upscale_factor` – при наличии выводится предупреждение.

3. **Взаимосвязи**  
   - `gt_size == network_g.upscale_factor * lq_size` – проверяется датасетом.
   - `network_g.upscale_factor == process_data.downscale_factor × 2` – рекомендуется для согласованности масштаба.

4. **Типы данных**  
   - `!!float` – явное указание числа с плавающей точкой (например, `lr: !!float 1e-4`).

---

## Пример полного конфигурационного файла

См. `configs/RAW_NAFNet_Final_100.yml` или `configs/NAFNet-100-02.yml`.

---

**Дата последнего обновления:** 2026-06-07  
**Версия:** 3.0 (соответствует коду после рефакторинга)
