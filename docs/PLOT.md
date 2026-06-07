## ✅ PLOT.md – версия 2.0 (актуальная)

Ниже приведён обновлённый `PLOT.md`, который учитывает новые столбцы в `train_metrics.csv` (`l1_loss`, `ffl_loss`, `delta_psnr`, `grad_var`, `tv_ratio`) и наличие `psnr` в `val_metrics.csv`. Также добавлены рекомендации по построению сравнительных графиков с baseline.

```markdown
# PLOT.md – Визуализация метрик обучения

Скрипт `plot_metrics.py` строит графики на основе CSV‑файлов, сгенерированных в процессе обучения (`train_metrics.csv`, `val_metrics.csv`). Для настройки внешнего вида и выбора данных используется конфигурационный YAML‑файл (например, `plot_metrics.yml`).

---

## 1. Доступные источники данных

| Источник | Путь по умолчанию | Содержание |
|----------|-------------------|-------------|
| `train_metrics.csv` | `experiments/{name}/train_metrics.csv` | Шаг, loss, psnr, lr, delta_psnr, grad_var, tv_ratio, vram_alloc_gb, vram_res_gb, l1_loss, ffl_loss |
| `val_metrics.csv` | `experiments/{name}/val_metrics.csv` | Эпоха, ssim, psnr |

---

## 2. Основные графики (пример конфигурации)

Ниже приведён полный `plot_metrics.yml`, охватывающий все полезные визуализации:

```yaml
experiment_dir: "experiments/RAW-NAFNet-Final-100"
output_dir: "experiments/RAW-NAFNet-Final-100/figures"

data_files:
  train_metrics: "experiments/RAW-NAFNet-Final-100/train_metrics.csv"
  val_metrics: "experiments/RAW-NAFNet-Final-100/val_metrics.csv"

baseline:
  psnr: 34.61
  ssim: 0.9640
  label: "Bilinear"

plots:
  - type: line
    title: "Кривая потерь (комбинированный loss)"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: loss
    smoothing: 0.9
    y_label: "Loss"
    save: loss_curve.png

  - type: multi_line
    title: "Компоненты лосса (L1 и FFL)"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: [l1_loss, ffl_loss]
    y_scale: log
    y_label: "Loss"
    legend: ["L1 loss", "FFL loss"]
    save: loss_components.png

  - type: line
    title: "PSNR на обучении"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: psnr
    smoothing: 0.95
    y_lim: [3, 40]
    y_label: "PSNR (dB)"
    save: psnr_curve.png

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
    title: "Learning Rate"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: lr
    y_scale: log
    y_label: "Learning rate"
    save: lr_schedule.png

  - type: multi_line
    title: "Метрики стабильности градиентов"
    data:
      source: csv
      file_ref: train_metrics
      x: step
      y: [delta_psnr, grad_var, tv_ratio]
    y_scale: log
    y_label: "Значение"
    legend: ["ΔPSNR", "Grad variance", "TV ratio"]
    save: stability_metrics.png

  - type: line
    title: "SSIM на валидации"
    data:
      source: val_csv
      x: epoch
      y: ssim
    y_lim: [0.0, 1.0]
    y_label: "SSIM"
    save: ssim_curve.png

  - type: line
    title: "PSNR на валидации"
    data:
      source: val_csv
      x: epoch
      y: psnr
    y_lim: [0, 40]
    y_label: "PSNR (dB)"
    save: val_psnr_curve.png

  - type: multi_line
    title: "Валидационные метрики"
    data:
      source: val_csv
      x: epoch
      y: [ssim, psnr]
    y_label: "Значение"
    legend: ["SSIM (×100)", "PSNR (dB)"]
    save: val_metrics.png
```

---

## 3. Типы графиков и их параметры

### `type: line`
Одиночная линия.
- `x` – колонка для оси X (обычно `step` или `epoch`).
- `y` – колонка для оси Y.
- `smoothing` – коэффициент экспоненциального сглаживания (1.0 – без сглаживания, 0.9 – сильное сглаживание).
- `y_lim` – минимальное и максимальное значение по Y (опционально).
- `y_scale` – `linear` или `log`.

### `type: multi_line`
Несколько линий на одном графике.
- `y` – список колонок.
- `legend` – подписи к линиям.

### `type: comparison`
Сравнение тренировочной кривой с горизонтальной линией (например, baseline).
- `baseline_value` – числовое значение.
- `baseline_label` – подпись.

---

## 4. Сглаживание (smoothing)

Используется экспоненциальное сглаживание (EMA):
```
smoothed[i] = α * raw[i] + (1-α) * smoothed[i-1]
```
- `α = 1.0` – оригинальные значения.
- `α = 0.95` – слабое сглаживание.
- `α = 0.9` – среднее.
- `α = 0.7` – сильное (для очень шумных кривых).

---

## 5. Особенности после рефакторинга (версия 2.0)

- **`l1_loss` и `ffl_loss`** – теперь логируются отдельно, что позволяет оценить вклад каждой компоненты.
- **`delta_psnr`** – изменение PSNR между шагами; около нуля говорит о плато.
- **`grad_var`** – дисперсия норм градиентов; резкий рост может сигнализировать о нестабильности.
- **`tv_ratio`** – отношение полной вариации выхода к таргету; значения >1 указывают на излишний шум, <0.5 – на чрезмерное сглаживание.
- **Валидационный PSNR** – теперь сохраняется в `val_metrics.csv` (ранее был только SSIM).

---

## 6. Запуск построения графиков

```bash
python plot_metrics.py -opt configs/plot_metrics.yml
```

Все графики сохранятся в папку, указанную в `output_dir` (обычно `experiments/{name}/figures/`).

---

## 7. Пример интерпретации

- **Кривая потерь (loss)** – должна монотонно убывать. Резкие скачки могут быть связаны с включением FFL на эпохе `ffl_start_epoch`.
- **PSNR** – на обучении должен расти и стабилизироваться. Если на валидации PSNR падает, а на обучении продолжает расти – переобучение.
- **Компоненты лосса** – полезно смотреть, не доминирует ли FFL (слишком большой вес) или не равен ли он нулю (если `ffl_start_epoch` ещё не наступил).
- **ΔPSNR** – если длительное время около нуля, стоит снизить LR или увеличить `ffl_weight`.
- **SSIM на валидации** – более стабильная метрика, чем PSNR. Для хорошей модели должен превышать 0.95.

---

## 8. Частые проблемы и решения

| Проблема | Решение |
|----------|---------|
| Нет файла `train_metrics.csv` | Убедитесь, что в конфиге `logger.log_csv: true` и обучение запущено. |
| График пустой или нет данных | Проверьте, что имена колонок в `y` совпадают с заголовками CSV (регистр важен). |
| Сглаживание искажает кривую | Уменьшите `smoothing` (ближе к 1.0) или отключите (`smoothing: 1.0`). |
| Не строятся графики из `val_csv` | Проверьте, что в `data.source` указано `val_csv` (не `csv`). |

---

## 9. Связь с другими документами

- [TRAINING.md](TRAINING.md) – описание метрик и их интерпретация.
- [PIPELINE.md](PIPELINE.md) – как запустить обучение и получить CSV.
- [CHANGES.md](CHANGES.md) – история изменений, включая добавление колонок в логи.

---

**Дата последнего обновления:** 2026-06-07  
**Версия:** 2.0 (полное соответствие текущей структуре логов)
