# UPDOWN_SCALE.md – Интеграция масштабов деградации и восстановления (downscale / upscale)

В данном документе объясняется, как согласованы коэффициенты уменьшения (`downscale_factor`) и увеличения (`upscale_factor`) в конвейере совместной демозаики и суперразрешения (JDSR). Правильная настройка этих параметров критически важна для физической корректности генерации данных и работы модели.

---

## 1. Общая формула сжатия

При генерации LQ из HQ применяются два последовательных уменьшения:

1. **Оптическое уменьшение** (симуляция ограниченного разрешения объектива) – коэффициент `downscale_factor`.  
   Из HQ‑патча размера `gt_size × gt_size` получаем изображение `(gt_size / downscale_factor) × (gt_size / downscale_factor)`.

2. **Субдискретизация, присущая маске Байера (RGGB)** – после наложения маски и упаковки в 4 подканала размер уменьшается ещё в 2 раза (каждый второй пиксель по горизонтали и вертикали выделяется в отдельный канал).

**Итоговый линейный размер LQ** (упакованного, 4 канала):

```
lq_size = gt_size // downscale_factor // 2
```

**Общее линейное сжатие от HQ до LQ:**

```
S = downscale_factor × 2
```

Пример: `downscale_factor = 2`, `gt_size = 512` → промежуточное 256×256 → LQ 128×128. Общее сжатие = 4.

---

## 2. Восстановление моделью (обёртка NAFNet)

Модель работает в **низком разрешении** на **4 каналах** (RGGB). Базовый NAFNet выдает тензор той же пространственной размерности (128×128) с 4 каналами. Затем применяется:

- Свёртка `Conv2d(4, out_channels * upscale_factor², 3, 1)`, которая увеличивает число каналов.
- `PixelShuffle(upscale_factor)` – перестраивает каналы в пространство.

Выход модели: RGB размером `(lq_size * upscale_factor) × (lq_size * upscale_factor) × 3`.

Для точного восстановления исходного HQ необходимо:

```
upscale_factor = S = downscale_factor × 2
```

Тогда размер выхода = `lq_size × (downscale_factor × 2) = (gt_size // downscale_factor // 2) × (downscale_factor × 2) = gt_size`.

---

## 3. Согласование размеров патчей в датасете

В конфигурационном файле задаются:

- `datasets.train.gt_size` – размер квадратного HQ‑патча.
- `datasets.train.lq_size` – ожидаемый размер LQ‑патча (до упаковки).

**Жёсткое условие** (проверяется `assert` в `CustomNEFPairDataset`):

```python
assert gt_size == upscale_factor * lq_size
```

Поскольку `upscale_factor` берётся из `network_g.upscale_factor`, а `lq_size` должен быть равен `gt_size // upscale_factor`. При генерации датасета `lq_size` вычисляется автоматически по формуле выше, но в конфиге его нужно указывать вручную, согласуя с этим равенством.

---

## 4. Примеры согласованных конфигураций

| `downscale_factor` | `upscale_factor` (S) | `gt_size` | `lq_size` (должен быть `gt_size / S`) | Комментарий |
|--------------------|----------------------|-----------|----------------------------------------|-------------|
| 1                  | 2                    | 256       | 128                                    | Только демозаика, без оптического сжатия |
| 2                  | 4                    | 512       | 128                                    | **Рекомендуемый** баланс качества и VRAM |
| 2                  | 4                    | 1024      | 256                                    | Для высокодетальных сцен (больше VRAM) |
| 4                  | 8                    | 1024      | 128                                    | Экстремальное сжатие (исследовательское) |

---

## 5. Проверка правильности настройки

После генерации датасета выполните проверочный скрипт:

```python
python -c "
from libraries.training_dataset_nef import CustomNEFPairDataset
from libraries.config import get_pipeline_config

opt = get_pipeline_config()
dataset = CustomNEFPairDataset(
    opt['path']['dataset_root'] + '/train/lq_inputs',
    opt['path']['dataset_root'] + '/train/hq_targets',
    opt=opt
)
lq, hq = dataset[0]
print(f'LQ shape: {lq.shape}')   # (4, lq_size, lq_size)
print(f'HQ shape: {hq.shape}')   # (3, gt_size, gt_size)
print(f'Соотношение: {hq.shape[1]} / {lq.shape[1]} = {hq.shape[1] / lq.shape[1]}')
# Ожидается: upscale_factor
"
```

Если вывод показывает правильное соотношение, значит настройки выполнены корректно.

---

## 6. Архитектурная схема (новая, правильная)

```
HQ (gt_size×gt_size×3)
       │
       ↓ (оптическое уменьшение downscale_factor, билинейная интерполяция)
       │
       ↓ (PSF‑размытие, шум, маска Байера)
       │
       ↓ (упаковка в 4 канала: extract_bayer_subchannels)
       │
LQ (lq_size×lq_size×4), где lq_size = gt_size // downscale_factor // 2
       │
       ↓ (NAFNet: вход 4 канала, выход 4 канала, низкое разрешение)
       │
       ↓ (Conv2d(4, 3*upscale_factor²) + PixelShuffle(upscale_factor))
       │
RGB (gt_size×gt_size×3) восстановленное
```

**Ключевые моменты:**
- NAFNet работает в низком разрешении (экономия памяти, скорость).
- PixelShuffle применяется **после** сети, а не до.
- Общий масштаб восстановления = `upscale_factor` должен равняться `downscale_factor × 2`.

---

## 7. Частые ошибки и их диагностика

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `AssertionError: gt_size != upscale_factor * lq_size` | В конфиге не согласованы `gt_size`, `lq_size` и `upscale_factor` | Установить `lq_size = gt_size // upscale_factor` |
| `Рассинхронизация размеров файлов для ...` | Датасет сгенерирован со старым `downscale_factor`, а в конфиге изменили `upscale_factor` | Перегенерировать датасет с `--clean-dataset` |
| Остаточный паттерн Байера после обучения | `upscale_factor` не равен `downscale_factor × 2` или модель обучалась на старой обёртке | Проверить конфиг, переобучить с правильной архитектурой |
| Out of Memory (OOM) | Модель работает в высоком разрешении из‑за неправильной обёртки | Использовать новую обёртку (PixelShuffle после NAFNet) |

---

## 8. Связь с другими компонентами

- **Генерация датасета** – `pipeline_generation_core.py` реализует формулу сжатия.
- **Датасет** – `training_dataset_nef.py` проверяет `gt_size == upscale_factor * lq_size`.
- **Модель** – `model_utils.py` (обёртка `NAFNetDemosaicSuperResolutionWrapper`) применяет PixelShuffle после сети.
- **Конфигурация** – `network_g.upscale_factor` – единый источник истины.

---

## 9. Резюме

**Одно главное правило:**

> **`network_g.upscale_factor = process_data.downscale_factor × 2`**  
> **`datasets.train.lq_size = datasets.train.gt_size // network_g.upscale_factor`**

Придерживаясь его, вы гарантированно получите физически корректную постановку задачи JDSR, оптимальное использование GPU и отсутствие артефактов масштабирования.

---

**Дата последнего обновления:** 2026-06-07  
**Версия:** 3.0 (соответствует новой архитектуре с PixelShuffle после NAFNet)
