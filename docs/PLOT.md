# PLOT.md – Построение графиков метрик обучения

Скрипт `plot_metrics.py` автоматизирует создание публикационно-готовых графиков на основе логов обучения и валидации (CSV‑файлов и текстовых логов). Он полностью конфигурируется через YAML, что позволяет гибко выбирать типы графиков, применять сглаживание, задавать подписи и сохранять результаты в папку эксперимента.

---

## 1. Запуск

```bash
./plot_metrics.py -opt options/plot/your_plot_config.yml
```

- `-opt` – **обязательный** путь к YAML‑конфигурационному файлу, в котором описаны все графики.

> **Примечание:** Скрипт не требует активации GPU и может выполняться на CPU.

---

## 2. Конфигурационный файл (YAML)

Файл конфигурации состоит из следующих секций:

### 2.1. Обязательные поля

| Поле | Тип | Описание | Пример |
|------|-----|----------|--------|
| `experiment_dir` | строка | Путь к папке эксперимента (содержит `train_metrics.csv`, `val_metrics.csv` и др.) | `"experiments/RAW-NAFNet-D600"` |
| `output_dir` | строка | Папка, куда будут сохранены графики (обычно внутри `experiment_dir/figures`) | `"experiments/RAW-NAFNet-D600/figures"` |
| `plots` | список | Перечень графиков (каждый – словарь с полями, см. ниже) | – |

### 2.2. Опциональная секция `data_files`

Позволяет задать короткие имена (алиасы) для путей к файлам, чтобы не повторять длинные пути в каждом графике.

```yaml
data_files:
  train_metrics: "experiments/RAW-NAFNet-D600/train_metrics.csv"
  val_metrics: "experiments/RAW-NAFNet-D600/val_metrics.csv"
  pipeline_log: "experiments/RAW-NAFNet-D600/pipeline.log"
```

В поле `data` графика можно ссылаться через `file_ref: train_metrics`.

### 2.3. Опциональная секция `baseline`

Задаёт значения для сравнения (например, PSNR классической демозаики), которые будут отображены в виде горизонтальной линии на графиках типа `comparison`.

| Поле | Тип | Описание | Пример |
|------|-----|----------|--------|
| `psnr` | число | Значение PSNR (dB) | `34.61` |
| `ssim` | число | Значение SSIM | `0.964` |
| `label` | строка | Подпись на графике | `"Bilinear"` |

### 2.4. Секция `plots` (список)

Каждый элемент списка описывает один график. Общие поля:

| Поле | Тип | Обязательное | Описание |
|------|-----|--------------|----------|
| `type` | строка | Да | Тип графика: `line`, `comparison`, `multi_line` |
| `title` | строка | Да | Заголовок графика |
| `save` | строка | Да | Имя файла для сохранения (PNG, будет помещён в `output_dir`) |
| `data` | словарь | Да | Источник данных (см. ниже) |
| `smoothing` | число | Нет | Коэффициент экспоненциального сглаживания (0..1); 1 – без сглаживания | `0.9` |
| `y_lim` | список [min, max] | Нет | Пределы по оси Y | `[0, 40]` |
| `y_scale` | строка | Нет | Тип шкалы: `"linear"` (по умолчанию) или `"log"` | `"log"` |
| `y_label` | строка | Нет | Подпись оси Y (по умолчанию – имя столбца или `"Value"`) | `"PSNR (dB)"` |
| `x_label` | строка | Нет | Подпись оси X (по умолчанию – имя столбца или `"Step"`) | `"Epoch"` |

#### 2.4.1. Поле `data` – описание источника данных

Возможные варианты:

- **`source: csv`** – чтение из CSV‑файла.
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

# CSV с несколькими столбцами (для multi_line)
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
  file_ref: pipeline_log
  pattern: "SSIM = (\\d+\\.\\d+)"
```

#### 2.4.2. Дополнительные поля для `type: comparison`

| Поле | Тип | Описание | Пример |
|------|-----|----------|--------|
| `baseline_value` | число | Значение горизонтальной линии (baseline) | `34.61` |
| `baseline_label` | строка | Подпись для baseline | `"Bilinear"` |

Если в секции `baseline` уже заданы значения, их можно переопределить для конкретного графика.

#### 2.4.3. Дополнительные поля для `type: multi_line`

| Поле | Тип | Описание | Пример |
|------|-----|----------|--------|
| `legend` | список строк | Подписи для каждой линии (должен быть той же длины, что и `y`) | `["ΔPSNR", "GradVar", "TV ratio"]` |

---

## 3. Пример полного конфигурационного файла

```yaml
experiment_dir: "experiments/RAW-NAFNet-D600"
output_dir: "experiments/RAW-NAFNet-D600/figures"

data_files:
  train_metrics: "experiments/RAW-NAFNet-D600/train_metrics.csv"
  val_metrics: "experiments/RAW-NAFNet-D600/val_metrics.csv"
  pipeline_log: "experiments/RAW-NAFNet-D600/pipeline.log"

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

---

## 4. Выходные данные

Скрипт создаёт папку `output_dir` (если её нет) и сохраняет все графики в формате PNG с разрешением 300 dpi. Имена файлов берутся из поля `save` каждого графика.

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

---

## 5. Замечания и рекомендации

- **Зависимости:** Для работы скрипта необходимы `matplotlib`, `pandas`, `numpy`, `scipy`. Они должны быть установлены в виртуальном окружении (см. `requirements.txt`).
- **Сглаживание:** Коэффициент `smoothing` (экспоненциальное скользящее среднее) помогает скрыть высокочастотные флуктуации. Для PSNR можно использовать `0.95`, для loss – `0.9`.
- **Масштаб оси Y:** Для метрик, изменяющихся на порядки (например, `grad_var`, `lr`), используйте `y_scale: log`.
- **Baseline:** Убедитесь, что указанные значения соответствуют реальным результатам классических методов на ваших тестовых данных (например, из `evaluate_baseline.py`).

---

## 6. Возможные проблемы и их решение

| Проблема | Причина | Решение |
|----------|---------|---------|
| `FileNotFoundError` | Неверный путь к файлу в `experiment_dir` или `data_files` | Проверьте наличие файлов и правильность путей |
| График пустой или нет данных | Некорректное имя столбца в `x` или `y` | Проверьте имена колонок в CSV (первая строка) |
| Регулярное выражение не извлекает числа | Неправильный `pattern` | Проверьте на примере строки из лога; используйте `(\\d+\\.\\d+)` для чисел с плавающей точкой |
| Ошибка импорта `matplotlib` | Библиотека не установлена | Выполните `pip install matplotlib pandas scipy` |

---

## 7. Интеграция с общим пайплайном

После завершения обучения (`run_training.py`) и, при необходимости, генерации baseline (`evaluate_baseline.py`), запустите `plot_metrics.py` для создания всех графиков, которые затем можно вставить в диссертацию или отчёт.

```bash
# 1. Обучить модель
./run_training.py -opt options/train/RAW_NAFNet_NikonD600.yml

# 2. Оценить baseline (билинейная, MHC)
./evaluate_baseline.py -opt options/train/RAW_NAFNet_NikonD600.yml

# 3. Построить графики
./plot_metrics.py -opt options/plot/plot_config.yml
```

--- 

**Дата составления:** 2026-06-06  
**Версия:** 1.0
