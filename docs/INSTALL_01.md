# Установка всех зависимостей
1. Сохранить:
```bash
pip freeze --local >> requirements.txt
```
2. Установить:
```bash
pip install -r requirements.txt
```
## Правильная установка TourchVision

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

## Проверка ядер CUDA

```bash
python3 -c "import torch; print('CUDA доступна:', torch.cuda.is_available()); print('Версия CUDA в Torch:', torch.version.cuda); print('Устройство:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
Должно быть что-то вроде

```text
CUDA доступна: True
Версия CUDA в Torch: 12.6
Устройство: NVIDIA GeForce GTX 1060 6GB
```

## Запуск

Нам нужно жестко прописать текущий путь в переменную окружения прямо перед командой запуска. Плюс, поскольку basicsr у вас дублируется (лежит и в корне проекта, и внутри NAFNet), безопаснее всего добавить оба пути в PYTHONPATH.

```bash
PYTHONPATH="$(pwd):$(pwd)/NAFNet" torchrun --nproc_per_node=1 NAFNet/basicsr/train.py -opt options/train/Tanahen/train_NAFNet_JPEG_1060.yml --launcher pytorch
```

Строго говоря, надо добавить пути модулей в окружение:
```bash
echo -e "$(pwd)\n$(pwd)/NAFNet" > .venv/lib/python3.12/site-packages/project_paths.pth
```

Или лучше воспользоваться файлом `pyproject.toml`:
```bash
pip install -e .
```

Теперь можно запускать
```bash
torchrun --nproc_per_node=1 NAFNet/basicsr/train.py -opt options/train/Tanahen/train_NAFNet_JPEG_1060.yml --launcher pytorch
```

## Результат обучения:

Смотреть здесь:
```bash
NAFNet/experiments/NAFNet-JPEG-1060/models/
```

Увидим что-то вроде:

```bash
(.venv) ➜  Photo-Restoration-PyTorch git:(main) ✗ torchrun --nproc_per_node=1 NAFNet/basicsr/train.py -opt options/train/Tanahen/train_NAFNet_JPEG_1060.yml --launcher pytorch                                  

init dist ..  pytorch
Path already exists. Rename it to /home/denis/Projects/Python/Photo-Restoration-PyTorch/NAFNet/experiments/NAFNet-JPEG-1060_archived_20260521_183720
Path already exists. Rename it to tb_logger/NAFNet-JPEG-1060_archived_20260521_183720
2026-05-21 18:37:20,722 INFO: 
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    
Version Information: 
	BasicSR: 1.2.0+386ca20
	PyTorch: 2.12.0+cu126
	TorchVision: 0.27.0+cu126
2026-05-21 18:37:20,722 INFO: 
  name: NAFNet-JPEG-1060
  model_type: ImageRestorationModel
  scale: 1
  num_gpu: 1
  manual_seed: 42
  path:[
    pretrain_network_g: None
    strict_load_g: True
    resume_state: None
    root: /home/denis/Projects/Python/Photo-Restoration-PyTorch/NAFNet
    experiments_root: /home/denis/Projects/Python/Photo-Restoration-PyTorch/NAFNet/experiments/NAFNet-JPEG-1060
    models: /home/denis/Projects/Python/Photo-Restoration-PyTorch/NAFNet/experiments/NAFNet-JPEG-1060/models
    training_states: /home/denis/Projects/Python/Photo-Restoration-PyTorch/NAFNet/experiments/NAFNet-JPEG-1060/training_states
    log: /home/denis/Projects/Python/Photo-Restoration-PyTorch/NAFNet/experiments/NAFNet-JPEG-1060
    visualization: /home/denis/Projects/Python/Photo-Restoration-PyTorch/NAFNet/experiments/NAFNet-JPEG-1060/visualization
  ]
  datasets:[
    train:[
      name: MyTrainDataset
      type: PairedImageDataset
      dataroot_gt: ./dataset_ready/train/hq_targets
      dataroot_lq: ./dataset_ready/train/lq_inputs
      filename_tmpl: {}
      io_backend:[
        type: disk
      ]
      gt_size: 128
      use_flip: True
      use_rot: True
      num_worker_per_gpu: 2
      batch_size_per_gpu: 2
      dataset_enlarge_ratio: 1
      phase: train
      scale: 1
    ]
    val:[
      name: MyTestDataset
      type: PairedImageDataset
      dataroot_gt: ./dataset_ready/test/hq_targets
      dataroot_lq: ./dataset_ready/test/lq_inputs
      io_backend:[
        type: disk
      ]
      phase: val
      scale: 1
    ]
  ]
  network_g:[
    type: NAFNet
    width: 32
    enc_blk_nums: [2, 2, 4, 8]
    middle_blk_num: 12
    dec_blk_nums: [2, 2, 2, 2]
  ]
  train:[
    optim_g:[
      type: AdamW
      lr: 0.001
      weight_decay: 0
      betas: [0.9, 0.9]
    ]
    scheduler:[
      type: TrueCosineAnnealingLR
      T_max: 200000
      eta_min: 1e-07
    ]
    total_iter: 200000
    warmup_iter: -1
    pixel_opt:[
      type: L1Loss
      loss_weight: 1.0
      reduction: mean
    ]
  ]
  logger:[
    print_freq: 50
    save_checkpoint_freq: 500
    use_tb_logger: True
    metrics:[
      psnr:[
        type: calculate_psnr
        crop_border: 0
        test_y_channel: False
      ]
      ssim:[
        type: calculate_ssim
        crop_border: 0
        test_y_channel: False
      ]
    ]
  ]
  is_train: True
  dist: True
  rank: 0
  world_size: 1

2026-05-21 18:37:20,798 INFO: Dataset PairedImageDataset - MyTrainDataset is created.
2026-05-21 18:37:20,798 INFO: Training statistics:
	Number of train images: 9
	Dataset enlarge ratio: 1
	Batch size per gpu: 2
	World size (gpu number): 1
	Require iter number per epoch: 5
	Total epochs: 40000; iters: 200000.
2026-05-21 18:37:20,798 INFO: Dataset PairedImageDataset - MyTestDataset is created.
2026-05-21 18:37:20,798 INFO: Number of val images/folders in MyTestDataset: 1
.. cosineannealingLR
2026-05-21 18:37:21,305 INFO: Model [ImageRestorationModel] is created.
2026-05-21 18:37:21,324 INFO: Start training from epoch: 0, iter: 0
2026-05-21 18:38:04,495 INFO: [NAFNe..][epoch: 12, iter:      50, lr:(1.000e-03,)] [eta: 1 day, 23:02:06, time (data): 0.250 (0.002)] l_pix: 2.7119e-02 
2026-05-21 18:38:41,036 INFO: [NAFNe..][epoch: 24, iter:     100, lr:(1.000e-03,)] [eta: 1 day, 19:50:01, time (data): 0.236 (0.000)] l_pix: 4.5983e-02 
2026-05-21 18:39:17,272 INFO: [NAFNe..][epoch: 37, iter:     150, lr:(1.000e-03,)] [eta: 1 day, 18:38:02, time (data): 0.225 (0.000)] l_pix: 2.9527e-02 
2026-05-21 18:39:52,532 INFO: [NAFNe..][epoch: 49, iter:     200, lr:(1.000e-03,)] [eta: 1 day, 17:45:22, time (data): 0.228 (0.000)] l_pix: 2.0285e-02 
2026-05-21 18:40:28,690 INFO: [NAFNe..][epoch: 62, iter:     250, lr:(1.000e-03,)] [eta: 1 day, 17:25:22, time (data): 0.239 (0.002)] l_pix: 3.7739e-02 
2026-05-21 18:41:04,737 INFO: [NAFNe..][epoch: 74, iter:     300, lr:(1.000e-03,)] [eta: 1 day, 17:10:35, time (data): 0.230 (0.000)] l_pix: 2.7482e-02 
2026-05-21 18:41:40,638 INFO: [NAFNe..][epoch: 87, iter:     350, lr:(1.000e-03,)] [eta: 1 day, 16:58:28, time (data): 0.241 (0.002)] l_pix: 1.5613e-02 
2026-05-21 18:42:16,818 INFO: [NAFNe..][epoch: 99, iter:     400, lr:(1.000e-03,)] [eta: 1 day, 16:51:32, time (data): 0.238 (0.002)] l_pix: 2.6680e-02 
2026-05-21 18:42:53,138 INFO: [NAFNe..][epoch:112, iter:     450, lr:(1.000e-03,)] [eta: 1 day, 16:47:02, time (data): 0.239 (0.000)] l_pix: 1.1550e-02 
2026-05-21 18:43:29,222 INFO: [NAFNe..][epoch:124, iter:     500, lr:(1.000e-03,)] [eta: 1 day, 16:41:45, time (data): 0.244 (0.000)] l_pix: 2.7272e-02 
2026-05-21 18:43:29,223 INFO: Saving models and training states.
2026-05-21 18:44:07,524 INFO: [NAFNe..][epoch:137, iter:     550, lr:(1.000e-03,)] [eta: 1 day, 16:50:41, time (data): 0.248 (0.000)] l_pix: 2.4528e-02 
2026-05-21 18:44:43,638 INFO: [NAFNe..][epoch:149, iter:     600, lr:(1.000e-03,)] [eta: 1 day, 16:45:56, time (data): 0.237 (0.000)] l_pix: 3.8085e-02 
2026-05-21 18:45:19,953 INFO: [NAFNe..][epoch:162, iter:     650, lr:(1.000e-03,)] [eta: 1 day, 16:42:51, time (data): 0.257 (0.000)] l_pix: 1.5430e-02 
2026-05-21 18:45:56,106 INFO: [NAFNe..][epoch:174, iter:     700, lr:(1.000e-03,)] [eta: 1 day, 16:39:21, time (data): 0.214 (0.000)] l_pix: 1.3798e-02 
2026-05-21 18:46:33,205 INFO: [NAFNe..][epoch:187, iter:     750, lr:(1.000e-03,)] [eta: 1 day, 16:40:25, time (data): 0.250 (0.000)] l_pix: 2.5429e-02 
2026-05-21 18:47:09,285 INFO: [NAFNe..][epoch:199, iter:     800, lr:(1.000e-03,)] [eta: 1 day, 16:37:03, time (data): 0.208 (0.000)] l_pix: 1.4257e-02 
2026-05-21 18:47:46,339 INFO: [NAFNe..][epoch:212, iter:     850, lr:(1.000e-03,)] [eta: 1 day, 16:37:48, time (data): 0.284 (0.000)] l_pix: 1.1374e-02 
2026-05-21 18:48:23,657 INFO: [NAFNe..][epoch:224, iter:     900, lr:(1.000e-03,)] [eta: 1 day, 16:39:23, time (data): 0.238 (0.000)] l_pix: 1.0866e-02 
2026-05-21 18:48:59,836 INFO: [NAFNe..][epoch:237, iter:     950, lr:(9.999e-04,)] [eta: 1 day, 16:36:46, time (data): 0.249 (0.000)] l_pix: 1.7613e-02 
2026-05-21 18:49:37,074 INFO: [NAFNe..][epoch:249, iter:   1,000, lr:(9.999e-04,)] [eta: 1 day, 16:37:50, time (data): 0.235 (0.000)] l_pix: 1.9727e-02 
2026-05-21 18:49:37,074 INFO: Saving models and training states.
2026-05-21 18:50:14,638 INFO: [NAFNe..][epoch:262, iter:   1,050, lr:(9.999e-04,)] [eta: 1 day, 16:39:47, time (data): 0.244 (0.000)] l_pix: 1.1256e-02 
2026-05-21 18:50:51,132 INFO: [NAFNe..][epoch:274, iter:   1,100, lr:(9.999e-04,)] [eta: 1 day, 16:38:17, time (data): 0.237 (0.002)] l_pix: 1.3202e-02 
2026-05-21 18:51:27,991 INFO: [NAFNe..][epoch:287, iter:   1,150, lr:(9.999e-04,)] [eta: 1 day, 16:37:55, time (data): 0.240 (0.000)] l_pix: 1.3245e-02 
2026-05-21 18:52:03,971 INFO: [NAFNe..][epoch:299, iter:   1,200, lr:(9.999e-04,)] [eta: 1 day, 16:35:05, time (data): 0.225 (0.000)] l_pix: 3.1245e-02 
2026-05-21 18:52:40,941 INFO: [NAFNe..][epoch:312, iter:   1,250, lr:(9.999e-04,)] [eta: 1 day, 16:35:04, time (data): 0.230 (0.000)] l_pix: 1.0527e-02 
2026-05-21 18:53:17,523 INFO: [NAFNe..][epoch:324, iter:   1,300, lr:(9.999e-04,)] [eta: 1 day, 16:34:01, time (data): 0.285 (0.002)] l_pix: 2.1004e-02 
2026-05-21 18:53:54,827 INFO: [NAFNe..][epoch:337, iter:   1,350, lr:(9.999e-04,)] [eta: 1 day, 16:34:45, time (data): 0.248 (0.000)] l_pix: 2.6729e-02 
2026-05-21 18:54:30,990 INFO: [NAFNe..][epoch:349, iter:   1,400, lr:(9.999e-04,)] [eta: 1 day, 16:32:43, time (data): 0.215 (0.000)] l_pix: 2.0300e-02 
2026-05-21 18:55:07,253 INFO: [NAFNe..][epoch:362, iter:   1,450, lr:(9.999e-04,)] [eta: 1 day, 16:30:59, time (data): 0.280 (0.000)] l_pix: 1.8211e-02 
2026-05-21 18:55:43,551 INFO: [NAFNe..][epoch:374, iter:   1,500, lr:(9.999e-04,)] [eta: 1 day, 16:29:25, time (data): 0.244 (0.000)] l_pix: 3.1804e-02 
2026-05-21 18:55:43,551 INFO: Saving models and training states.
2026-05-21 18:56:21,539 INFO: [NAFNe..][epoch:387, iter:   1,550, lr:(9.999e-04,)] [eta: 1 day, 16:31:31, time (data): 0.269 (0.002)] l_pix: 2.0859e-02 
2026-05-21 18:56:58,389 INFO: [NAFNe..][epoch:399, iter:   1,600, lr:(9.998e-04,)] [eta: 1 day, 16:31:06, time (data): 0.243 (0.002)] l_pix: 2.7065e-02 
2026-05-21 18:57:35,151 INFO: [NAFNe..][epoch:412, iter:   1,650, lr:(9.998e-04,)] [eta: 1 day, 16:30:29, time (data): 0.261 (0.002)] l_pix: 3.2359e-02 
2026-05-21 18:58:11,335 INFO: [NAFNe..][epoch:424, iter:   1,700, lr:(9.998e-04,)] [eta: 1 day, 16:28:45, time (data): 0.229 (0.002)] l_pix: 1.6059e-02 
2026-05-21 18:58:47,668 INFO: [NAFNe..][epoch:437, iter:   1,750, lr:(9.998e-04,)] [eta: 1 day, 16:27:22, time (data): 0.244 (0.000)] l_pix: 1.3948e-02 
2026-05-21 18:59:23,525 INFO: [NAFNe..][epoch:449, iter:   1,800, lr:(9.998e-04,)] [eta: 1 day, 16:25:09, time (data): 0.236 (0.002)] l_pix: 1.9730e-02 
```
### Расшифровка:

- 2026-05-21 18:59:23,525 INFO: — Точная дата и время, когда была зафиксирована эта итерация.
- [epoch:449, iter: 1,800] — Текущий прогресс обучения:
- epoch: 449 — Сеть посмотрела весь ваш датасет из 9 картинок уже 449 раз.
- iter: 1,800 — Видеокарта сделала ровно 1800 шагов (итераций) вперед-назад по градиентам.
- lr:(9.998e-04,) — Текущий Learning Rate (скорость обучения) [ссылка на конфиг из истории]. В начале стояло значение 0.001 (1e-3), а сейчас планировщик - CosineAnnealing начал плавно и мизерно снижать скорость, чтобы сеть точнее подстраивалась под текстуры [ссылка на конфиг из истории].
- [eta: 1 day, 16:25:09] — Estimated Time of Arrival (оставшееся время). С текущей скоростью вашей GTX 1060 до полного окончания обучения (всех 200 000 итераций) осталось считать 1 день, 16 часов и 25 минут [ссылка на конфиг из истории].
- time (data): 0.236 (0.002) — Скорость работы в секундах:
- 0.236 — Столько времени (в секундах) видеокарта тратит на один шаг (обработку одного батча из 2 картинок).
(0.002) — Столько времени процессор тратит на то, чтобы считать картинки с диска и подготовить их для GPU. Это отличный показатель, процессор не тормозит видеокарту.
- l_pix: 1.9730e-02 — Самый главный показатель качества (L1 Loss/Pixel Loss) [ссылка на конфиг из истории]. Это средняя ошибка между восстановленной картинкой и идеальным PNG-оригиналом [ссылка на конфиг из истории]. Она равна 0.0197. Чем меньше это число, тем идеальнее работает сеть. В самом начале на iter: 0 оно было значительно выше.