# README: `plot_metrics.py` – конфигурируемый графопостроитель

## 1. Назначение

`plot_metrics.py` – это утилита для **автоматического построения графиков** на основе логов обучения и валидации. Она читает CSV-файлы с метриками (например, `train_metrics.csv`, `val_metrics.csv`) и текстовые логи (`pipeline.log`), и создаёт профессиональные иллюстрации для диссертации (кривые обучения, сравнение с baseline, вспомогательные метрики).

Скрипт полностью конфигурируется через YAML-файл, что позволяет гибко выбирать, какие графики строить, применять сглаживание, задавать пределы осей и сохранять результат в папку эксперимента.

## 2. Ключи командной строки

| Ключ | Тип | Обязательный | Описание |
|------|-----|--------------|-----------|
| `-opt PATH` | строка | **Да** | Путь к конфигурационному YAML-файлу (например, `options/plot/RAW_NAFNet_NikonD600.yml`). |

**Пример запуска:**  
```bash
./plot_metrics.py -opt options/plot/RAW_NAFNet_NikonD600.yml
```

## 3. Конфигурационный YAML-файл

Файл конфигурации состоит из нескольких обязательных и опциональных секций.

### 3.1. Обязательные секции

| Ключ | Тип | Описание | Пример |
|------|-----|----------|--------|
| `experiment_dir` | строка | Путь к папке эксперимента (содержит `train_metrics.csv`, `val_metrics.csv` и др.). | `"experiments/RAW-NAFNet-D600"` |
| `output_dir` | строка | Папка, куда будут сохранены графики (обычно внутри `experiment_dir/figures`). | `"experiments/RAW-NAFNet-D600/figures"` |
| `plots` | список | Перечень графиков для построения (описание каждого – ниже). | – |

### 3.2. Опциональная секция `data_files`

Используется для задания коротких имён (алиасов) файлов, чтобы не повторять длинные пути. Каждый алиас сопоставляется с путём к файлу (абсолютным или относительным). В дальнейшем в описании графика можно использовать `file_ref: алиас`.

Пример:
```yaml
data_files:
  train_metrics: "experiments/RAW-NAFNet-D600/train_metrics.csv"
  val_metrics: "experiments/RAW-NAFNet-D600/val_metrics.csv"
```

### 3.3. Опциональная секция `baseline`

Задаёт значения для сравнения (например, PSNR билинейной интерполяции). Используется в графиках типа `comparison`.

| Ключ | Тип | Описание | Пример |
|------|-----|----------|--------|
| `psnr` | число | Значение PSNR baseline (dB). | `34.61` |
| `ssim` | число | Значение SSIM baseline. | `0.964` |
| `label` | строка | Подпись на графике. | `"Bilinear"` |

### 3.4. Секция `plots` (список)

Каждый элемент списка описывает один график. Общие поля:

| Ключ | Тип | Обязательный | Описание |
|------|-----|--------------|-----------|
| `type` | строка | Да | Тип графика: `line`, `comparison`, `multi_line`. |
| `title` | строка | Да | Заголовок графика. |
| `save` | строка | Да | Имя файла для сохранения (будет помещён в `output_dir`). |
| `data` | словарь | Да | Источник данных (см. ниже). |
| `smoothing` | число | Нет | Коэффициент экспоненциального сглаживания (0..1). 1 – без сглаживания. | `0.9` |
| `y_lim` | список из 2 чисел | Нет | Пределы по оси Y `[min, max]`. | `[0, 40]` |
| `y_scale` | строка | Нет | `"linear"` (по умолчанию) или `"log"`. | `"log"` |
| `y_label` | строка | Нет | Подпись оси Y (по умолчанию – имя столбца или `"Value"`). | `"PSNR (dB)"` |
| `x_label` | строка | Нет | Подпись оси X (по умолчанию – имя столбца или `"Step"`). | `"Epoch"` |

#### 3.4.1. Поле `data`

Определяет, откуда брать данные. Возможные варианты:

- **`source: csv`** – чтение из CSV-файла.
  - Обязательные поля: `file` (путь) или `file_ref` (алиас из `data_files`), `x` (имя столбца для оси X), `y` (имя столбца для оси Y).
  - Для `type: multi_line` поле `y` должно быть списком строк (несколько столбцов).
- **`source: val_csv`** – автоматический поиск `val_metrics.csv` в папке `experiment_dir`. Остальные поля аналогичны `csv`.
- **`source: log`** – извлечение данных из текстового лога с помощью регулярного выражения.
  - Обязательные поля: `file` (путь) или `file_ref`, `pattern` (регулярное выражение, из которого извлекается число). Ось X – номер строки (эпоха).

Примеры:

```yaml
# CSV с одним столбцом
data:
  source: csv
  file_ref: train_metrics
  x: step
  y: loss

# CSV с несколькими столбцами (multi_line)
data:
  source: csv
  file_ref: train_metrics
  x: step
  y: [delta_psnr, grad_var, tv_ratio]

# Автоматический поиск val_metrics.csv
data:
  source: val_csv
  x: epoch
  y: ssim

# Из лога
data:
  source: log
  file: experiments/RAW-NAFNet-D600/pipeline.log
  pattern: "SSIM = (\\d+\\.\\d+)"
```

#### 3.4.2. Дополнительные поля для `type: comparison`

| Ключ | Тип | Описание | Пример |
|------|-----|----------|--------|
| `baseline_value` | число | Значение горизонтальной линии (baseline). | `34.61` |
| `baseline_label` | строка | Подпись для baseline. | `"Bilinear"` |

#### 3.4.3. Дополнительные поля для `type: multi_line`

| Ключ | Тип | Описание | Пример |
|------|-----|----------|--------|
| `legend` | список строк | Подписи для каждой линии (должен быть той же длины, что и `y`). | `["ΔPSNR", "GradVar", "TV ratio"]` |

## 4. Пример конфигурационного файла

```yaml
experiment_dir: "experiments/RAW-NAFNet-D600"
output_dir: "experiments/RAW-NAFNet-D600/figures"

data_files:
  train_metrics: "experiments/RAW-NAFNet-D600/train_metrics.csv"
  val_metrics: "experiments/RAW-NAFNet-D600/val_metrics.csv"

baseline:
  psnr: 34.61
  ssim: 0.964
  label: "Bilinear"

plots:
  - type: line
    title: "Кривая потерь (Loss)"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: loss
    smoothing: 0.9
    y_label: "Loss"
    save: loss_curve.png

  - type: line
    title: "PSNR на обучении"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: psnr
    smoothing: 0.95
    y_lim: [0, 45]
    y_label: "PSNR (dB)"
    save: psnr_curve.png

  - type: line
    title: "Learning Rate"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: lr
    y_scale: log
    y_label: "Learning rate"
    save: lr_schedule.png

  - type: comparison
    title: "Сравнение PSNR: NAFNet vs Bilinear"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: psnr
    baseline_value: 34.61
    baseline_label: "Bilinear"
    y_label: "PSNR (dB)"
    save: psnr_comparison.png

  - type: line
    title: "SSIM на валидации"
    data:
      source: val_csv
      x: epoch
      y: ssim
    y_lim: [0.5, 1.0]
    y_label: "SSIM"
    save: ssim_curve.png

  - type: multi_line
    title: "Дополнительные метрики"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: [delta_psnr, grad_var, tv_ratio]
    y_scale: log
    y_label: "Значение"
    legend: ["ΔPSNR", "Grad variance", "TV ratio"]
    save: aux_metrics.png
```

## 5. Что генерирует скрипт и куда сохраняет

Скрипт создаёт папку `output_dir` (если её нет) и сохраняет в неё все указанные графики в формате PNG с разрешением 300 dpi (настраивается в коде). Имена файлов задаются в поле `save` каждого графика.

**Пример выходной структуры:**
```
experiments/RAW-NAFNet-D600/figures/
├── loss_curve.png
├── psnr_curve.png
├── lr_schedule.png
├── psnr_comparison.png
├── ssim_curve.png
└── aux_metrics.png
```

## 6. Вывод

`plot_metrics.py` – это универсальный инструмент для визуализации результатов обучения. Он:

- Не требует изменений при перезапуске экспериментов – достаточно указать путь к папке эксперимента.
- Поддерживает различные типы графиков и источники данных (CSV, текстовые логи).
- Позволяет настраивать сглаживание, масштаб осей, подписи.
- Создаёт публикационно-готовые изображения (высокое разрешение, чистый стиль).

Использование этого скрипта гарантирует единообразие иллюстраций в диссертации и экономит время на ручном построении графиков.