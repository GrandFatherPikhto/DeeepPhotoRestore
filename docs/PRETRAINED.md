# PRETRAINED.md – Инференс предобученных моделей NAFNet

Скрипт `inference_pretrained.py` применяет предобученные модели NAFNet (например, с датасетов SIDD, GoPro, REDS) к **вашим 4‑канальным RGGB‑файлам** или к обычным RGB‑изображениям. Он выполняет билинейную демозаику (если нужно), обрабатывает изображение целиком или по патчам с перекрытием, вычисляет метрики (PSNR, SSIM) при наличии эталона и сохраняет результат.

---

## 1. Назначение

- **Сравнение** вашей модели, обученной на малом датасете, с современными предобученными моделями.
- **Оценка** влияния архитектуры (`width=32` или `64`) и задачи (шумоподавление, деблюр) на качество восстановления.
- **Получение** численных метрик и визуальных результатов для диссертации.

---

## 2. Запуск

```bash
./inference_pretrained.py -opt path/to/config.yml
```

- `-opt` – **обязательный** путь к YAML‑конфигурационному файлу.

> **Примечание:** Скрипт не использует ключи `--resume` или `--clean-*`. Все настройки задаются в YAML.

---

## 3. Конфигурационный файл (YAML)

Пример минимального конфига (для SIDD `width=32`):

```yaml
# inference_sidd32.yml
name: pretrained_sidd32

paths:
  input_tiff: "datasets/nef_nafnet/test/lq_inputs/DSC_0413_bayer.tiff"
  gt_png: "datasets/nef_nafnet/test/hq_targets/DSC_0413.png"   # опционально
  output_dir: "experiments/pretrained_comparison/results"
  checkpoint: "checkpoints/nafnet_sidd_width32.pth"

network_g:
  type: NAFNet                # всегда NAFNet
  num_in_ch: 3                # предобученные модели ожидают RGB (3 канала)
  num_out_ch: 3
  width: 32                   # должно совпадать с моделью (32 или 64)
  middle_blk_num: 12
  enc_blk_nums: [2,2,4,8]
  dec_blk_nums: [2,2,2,2]

processing:
  bilinear_demosaic: true     # преобразовать 4‑канальный TIFF в RGB
  mode: "patch"               # "full" или "patch"
  patch_size: 256             # размер патча (для режима patch)
  overlap: 32                 # перекрытие между патчами (пиксели)

pipeline_logger:
  log_file: "experiments/pretrained_comparison/pipeline.log"
```

### 3.1. Секция `paths`

| Параметр | Обязательный | Описание |
|----------|-------------|----------|
| `input_tiff` | Да | Путь к 4‑канальному TIFF (RGGB) или RGB‑изображению (если `bilinear_demosaic: false`). |
| `gt_png` | Нет | Путь к эталонному RGB (PNG). Если указан, вычисляются PSNR и SSIM. |
| `output_dir` | Да | Папка для сохранения результата. |
| `checkpoint` | Да | Путь к файлу предобученной модели (.pth). |

### 3.2. Секция `network_g`

Должна точно соответствовать архитектуре загружаемой модели (особенно `width` и количество блоков). Предобученные модели NAFNet обычно имеют:

- `num_in_ch = 3`
- `num_out_ch = 3`
- `width = 32` или `64`
- `middle_blk_num = 12`
- `enc_blk_nums = [2,2,4,8]`
- `dec_blk_nums = [2,2,2,2]`

### 3.3. Секция `processing`

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `bilinear_demosaic` | `true` | Если `true`, входной TIFF (4 канала) преобразуется в RGB через билинейную демозаику. Если `false`, файл читается как обычное RGB (например, для JPEG). |
| `mode` | `"patch"` | Режим обработки: `"full"` – всё изображение целиком (требует много VRAM), `"patch"` – разбиение на патчи с перекрытием (экономит память). |
| `patch_size` | `256` | Размер квадратного патча (только для `mode="patch"`). |
| `overlap` | `32` | Перекрытие между соседними патчами (пиксели). Уменьшает артефакты на стыках. |

---

## 4. Режимы обработки

- **`mode: "full"`** – модель получает всё изображение за один проход. Подходит для небольших изображений (≤512×512) или при достаточном объёме VRAM.
- **`mode: "patch"`** – изображение разбивается на патчи заданного размера с перекрытием. Каждый патч обрабатывается отдельно, результаты усредняются в области перекрытия. Рекомендуется для больших изображений (например, Full HD и выше).

---

## 5. Где взять предобученные модели

Все модели доступны в официальном репозитории [megvii‑research/NAFNet](https://github.com/megvii-research/NAFNet) (Google Drive) или на Hugging Face ([nyanko7/nafnet-models](https://huggingface.co/nyanko7/nafnet-models)).

| Модель | Задача | PSNR / SSIM (на SIDD/GoPro) | Вес | Ссылка (пример) |
|--------|--------|----------------------------|-----|----------------|
| `NAFNet-SIDD-width32` | Шумоподавление | 39.97 / 0.9599 | ~117 МБ | [Google Drive](https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ) |
| `NAFNet-SIDD-width64` | Шумоподавление | 40.30 / 0.9614 | ~464 МБ | [Google Drive](https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR) |
| `NAFNet-GoPro-width32` | Деблюр | 32.87 / 0.9606 | ~68 МБ | [Google Drive](https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj) |
| `NAFNet-GoPro-width64` | Деблюр | 33.71 / 0.9668 | ~272 МБ | [Google Drive](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC) |
| `NAFNet-REDS-width64` | Деблюр | – | ~272 МБ | [Google Drive](https://drive.google.com/file/d/1bBdblOGvFfLXgPjATAnPxJqjCzTgTvl6) |

> **Важно:** Сохраняйте скачанные файлы в папку, указанную в `paths.checkpoint` (например, `checkpoints/nafnet_sidd_width32.pth`).

---

## 6. Пример запуска

```bash
# Для SIDD-width32
./inference_pretrained.py -opt options/inference/pretrained_sidd32.yml

# Для GoPro-width64
./inference_pretrained.py -opt options/inference/pretrained_gopro64.yml
```

### Пример вывода в консоль:
```
2026-06-06 12:00:15.123 [INFO] Устройство: cuda
2026-06-06 12:00:15.456 [INFO] Модель загружена из checkpoints/nafnet_sidd_width32.pth
2026-06-06 12:00:15.500 [INFO] Входное RGB размером (1024, 1024)
2026-06-06 12:00:15.512 [INFO] Режим патчей: patch_size=256, overlap=32
2026-06-06 12:00:18.234 [INFO] Результат сохранён в experiments/pretrained_comparison/results/DSC_0413_pretrained.png
2026-06-06 12:00:18.245 [INFO] PSNR = 28.34 dB, SSIM = 0.9123
```

---

## 7. Возможные проблемы и их решение

| Проблема | Причина | Решение |
|----------|---------|---------|
| `ModuleNotFoundError: No module named 'basicsr'` | Не установлена библиотека `basicsr` из субмодуля NAFNet | Установите NAFNet как описано в [INSTALL.md](INSTALL.md) (раздел 6). |
| `RuntimeError: Error(s) in loading state_dict` | Несовпадение архитектуры: `width` или количество блоков не соответствуют чекпоинту | Проверьте `network_g` в конфиге – должно быть точно как в оригинальной модели. |
| Ошибка памяти CUDA (OOM) при `mode: "full"` | Изображение слишком большое | Используйте `mode: "patch"` с меньшим `patch_size` (например, 256) и `overlap=32`. |
| Некорректные метрики PSNR/SSIM | Размеры выхода модели и GT не совпадают | Скрипт автоматически ресайзит GT к размеру выхода. Если ошибка всё же есть, проверьте `gt_png` и `bilinear_demosaic`. |
| Демозаика даёт артефакты | Билинейная интерполяция – не лучший метод | Для предобученных моделей это неизбежно, но можно попробовать более качественную демозаику (например, MHC из `evaluate_baseline.py`). |

---

## 8. Сравнение с вашей моделью

Если вы уже обучили свою модель (на 4 каналах RGGB), используйте для инференса скрипт `inference.py` (если он есть) или доработайте `inference_pretrained.py`, заменив предобученные веса на свои. Для честного сравнения:

- Подавайте на вход **один и тот же** 4‑канальный TIFF.
- Для предобученных моделей сначала делайте билинейную демозаику (`bilinear_demosaic: true`).
- Для вашей модели подавайте 4‑канальный тензор напрямую (без демозаики).

Результаты сохраняйте в отдельные папки и вычисляйте метрики. Это даст наглядное сравнение в диссертации.

---

## 9. Дополнительные ресурсы

- Официальный репозиторий NAFNet: [github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)
- Hugging Face зеркало: [nyanko7/nafnet-models](https://huggingface.co/nyanko7/nafnet-models)
- Документация по установке: [INSTALL.md](INSTALL.md)
- Документация по обучению: [TRAINING.md](TRAINING.md)

---

**Дата составления:** 2026-06-06  
**Версия:** 1.0
