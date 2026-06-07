# TODO

**Что хорошо в этой версии**

`training_validator.py` — чистый, `self.logger` везде, PSNR добавлен, геометрия через `upscale_factor`. `training_losses.py` — `ffl_start_epoch` и логарифмический вариант — грамотно. `training_utils.py` — `compute_metrics` передаёт `current_epoch` в `CombinedLoss`. `training_dataset_nef.py` — assert на геометрию, единый источник `upscale_factor` из `network_g`.

---

**Проблема с мылом**

Моя рекомендация из прошлого ответа актуальна. Но!.

В `run_training.py` сейчас:

```python
mean_ssim, mean_psnr = validator.run_validation(model, epoch, device)
train_logger.log_validation_metrics(epoch, mean_ssim)  # psnr не передаётся!
```

Это так?

`log_validation_metrics` ожидает `(epoch, ssim, psnr_val)` — три аргумента. Передаётся два — `psnr` не логируется в CSV и TensorBoard. Это не причина мыла, но данные для диссертации теряются.

**`video_degradation.py` — похоже на баг:**

```python
hq = frame_rgb.astype(np.float32) / 255.0   # float [0,1]
lq_packed, _ = generate_lq_from_hq(hq, config)  # ожидает uint8!
```

Внутри `generate_lq_from_hq` происходит `np.clip(hq_rgb, 0, 255).astype(np.uint8)` — из float [0,1] получится изображение почти полностью чёрное (все значения 0). Видеоинференс будет давать мусор. Исправь одной строкой:

```python
hq = frame_rgb  # уже uint8, не делить на 255
```

**`pipeline_prepare.py` — `dataset_root` из плоского ключа:**

```python
dataset_root = config.get("dataset_root", "datasets/nef_nafnet")
```

Но после твоих правок конфига он лежит в `config['path']['dataset_root']`. Генерация будет работать с дефолтным путём, игнорируя YAML. Исправь:

```python
dataset_root = config.get('path', {}).get('dataset_root', 'datasets/nef_nafnet')
```

**`pipeline_data.py` — `phase = "val"` для `is_train=False`** Папки `val/` нет. стоит так сделать?

---

**Три действия:**

1. Исправить архитектуру: NAFNet в LR-пространстве, PixelShuffle после.
2. Убрать заморозку весов из `run_training.py` если она ещё есть.
3. Пересгенерировать датасет с `--clean-dataset` — старые файлы могли быть сгенерированы с неверной нормировкой float.