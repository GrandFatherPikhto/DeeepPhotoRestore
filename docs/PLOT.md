# PLOT.md – Визуализация метрик обучения

Скрипт `plot_metrics.py` строит графики на основе CSV‑файлов, сгенерированных в процессе обучения (`train_metrics.csv`, `val_metrics.csv`). Для настройки внешнего вида и выбора данных используется конфигурационный YAML‑файл (например, `plot_metrics.yml`).  
После рефакторинга `plot_metrics.py` разбит на модули: `libraries/plot_utils.py` (утилиты, загрузка, подстановка `{name}`) и `libraries/plot_functions.py` (функции построения графиков). Корневой скрипт стал компактнее.

---

## 1. Доступные источники данных

| Источник | Путь по умолчанию | Содержание |
|----------|-------------------|-------------|
| `train_metrics.csv` | `experiments/{name}/train_metrics.csv` | step, loss, psnr, lr, delta_psnr, grad_var, tv_ratio, vram_alloc_gb, vram_res_gb, l1_loss, ffl_loss |
| `val_metrics.csv` | `experiments/{name}/val_metrics.csv` | epoch, psnr, ssim |

---

## 2. Основные графики (пример конфигурации)

Ниже приведён полный `plot_metrics.yml`, охватывающий все полезные визуализации с учётом новых колонок.

```yaml
name: NAFNet-100-02                     # опционально, для подстановки {name}
experiment_dir: "experiments/{name}"    # или явный путь, например "experiments/RAW-NAFNet-Final-100"
output_dir: "experiments/{name}/figures"

data_files:
  train_metrics: "experiments/{name}/train_metrics.csv"
  val_metrics: "experiments/{name}/val_metrics.csv"

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
    baseline_value: 34.61               # значение PSNR билинейной демозаики на вашем тесте
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
      y: [psnr, ssim]
    y_label: "Значение"
    legend: ["PSNR (dB)", "SSIM (×100)"]
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
- `baseline_value` – числовое значение (PSNR или SSIM).
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

- **Поддержка `{name}`** – в конфиге `plot_metrics.yml` можно использовать `{name}` в путях (`experiment_dir`, `output_dir`, `data_files.*`). При запуске `plot_metrics.py` нужно либо указать `name` в YAML, либо передать `--name` в командной строке. Подстановка работает рекурсивно.
- **Новые колонки** – `l1_loss`, `ffl_loss`, `delta_psnr`, `grad_var`, `tv_ratio` – все они доступны для построения.
- **Валидационный PSNR** – теперь сохраняется в `val_metrics.csv` (колонка `psnr`).
- **Модульность** – функции загрузки и построения вынесены в `libraries/plot_utils.py` и `libraries/plot_functions.py`. Это упрощает поддержку и переиспользование.

---

## 6. Запуск построения графиков

```bash
# Если в plot_metrics.yml есть поле name
python plot_metrics.py -opt options/plot/your_plot_config.yml

# Если нужно переопределить имя эксперимента
python plot_metrics.py -opt options/plot/your_plot_config.yml --name NAFNet-100-02
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
| Нет файла `train_metrics.csv` | Убедитесь, что в конфиге обучения `logger.log_csv: true` и обучение запущено. |
| График пустой или нет данных | Проверьте, что имена колонок в `y` совпадают с заголовками CSV (регистр важен). В `val_metrics.csv` колонки `epoch`, `psnr`, `ssim`. |
| Сглаживание искажает кривую | Уменьшите `smoothing` (ближе к 1.0) или отключите (`smoothing: 1.0`). |
| Не строятся графики из `val_csv` | Проверьте, что в `data.source` указано `val_csv` (не `csv`). |
| `{name}` не заменяется | Убедитесь, что в конфиге `plot_metrics.yml` есть поле `name` или передан аргумент `--name`. |
| Ошибка `KeyError: 'experiment_dir'` | Добавьте в конфиг обязательную секцию `experiment_dir`. |

---

## 9. Расширение (добавление новых типов графиков)

Чтобы добавить новый тип графика:
1. Реализуйте функцию построения в `libraries/plot_functions.py` (например, `plot_histogram`).
2. В `plot_metrics.py` добавьте ветку `elif plot_type == 'histogram':` в блоке обработки.
3. Обновите документацию.

---

## 10. Связь с другими документами

- [TRAINING.md](TRAINING.md) – описание метрик и их интерпретация.
- [PIPELINE.md](PIPELINE.md) – как запустить обучение и получить CSV.
- [CHANGES.md](CHANGES.md) – история изменений, включая рефакторинг `plot_metrics.py`.
- [CONFIG.md](CONFIG.md) – описание параметров конфигурации графиков.

---

**Дата последнего обновления:** 2026-06-07  
**Версия:** 2.0 (поддержка подстановки {name}, модульная структура, учёт новых колонок)
