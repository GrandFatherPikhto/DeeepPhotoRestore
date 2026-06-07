## === run_pipeline.py (part 1) ===
import os
import sys
import shutil
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from libraries.config import get_pipeline_config
from libraries.logger import setup_logger, get_logger
from libraries.device import get_torch_device
from libraries.pipeline_prepare import ensure_dataset_ready
from libraries.training_dataset_nef import CustomNEFPairDataset
from libraries.model_utils import create_nafnet_model
from libraries.pipeline_visuals import run_visual_control
from libraries.pipeline_smoke import run_smoke_test
def main():
    config = get_pipeline_config()
    opt_path = config.get("opt_path")
    if not opt_path:
        raise ValueError("Не передан параметр -opt")
    logger = setup_logger(config.get("pipeline_logger", {}).get("log_file", "pipeline.log"))
    logger.info("🚀 СТАРТ КОНВЕЙЕРА (используется CustomNEFPairDataset)")
    device = get_torch_device()
    try:
        clean_dataset_flag = config.get("clean_dataset", False)
        ensure_dataset_ready(config, clean_dataset=clean_dataset_flag)
        dataset_root = config['path']['dataset_root']
        train_lq_dir = os.path.join(dataset_root, 'train', 'lq_inputs')
        train_hq_dir = os.path.join(dataset_root, 'train', 'hq_targets')
        if not os.path.exists(train_lq_dir) or not os.path.exists(train_hq_dir):
            logger.error(f"Папки датасета не найдены: {train_lq_dir} или {train_hq_dir}")
            sys.exit(1)
        config['datasets'] = config.get('datasets', {})
        config['datasets']['train'] = config['datasets'].get('train', {})
        config['datasets']['train']['use_flip'] = False
        config['datasets']['train']['use_rot'] = False
        dataset = CustomNEFPairDataset(train_lq_dir, train_hq_dir, opt=config)
        logger.info(f"Датасет загружен, {len(dataset)} пар LQ/HQ")
        run_visual_control(dataset, config)
        model = create_nafnet_model(config, device)
        run_smoke_test(model, dataset, config, device)
        logger.info("🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
    except Exception as e:
        logger.critical(f"💥 АВАРИЙНОЕ ЗАВЕРШЕНИЕ: {e}", exc_info=True)
        sys.exit(1)
if __name__ == "__main__":
    main()

## === run_training.py (part 1) ===
import sys
import torch
import torch.nn as nn
from libraries.training_utils import (
    setup_experiment, create_train_loader, create_optimizer_and_scheduler,
    cleanup_experiment, get_loss_criterion, compute_metrics, log_progress
)
from libraries.model_utils import create_nafnet_model
from libraries.training_logger import TrainingLogger
from libraries.training_validator import VisualValidator
from libraries.training_checkpoint import get_checkpoint_path, load_checkpoint, save_checkpoint
def main():
    opt, save_dir, device, logger = setup_experiment()
    ignore_resume = cleanup_experiment(opt, save_dir, logger)
    train_loader = create_train_loader(opt)
    pretrained_path = opt.get('path', {}).get('pretrain_network_g', None)
    model = create_nafnet_model(opt, device, pretrained_path=pretrained_path)
    total_steps = len(train_loader) * opt['train']['num_epochs']       
    optimizer, scheduler = create_optimizer_and_scheduler(model, opt, total_training_steps=total_steps)
    train_logger = TrainingLogger(opt, save_dir)
    validator = VisualValidator(opt)
    checkpoint_path = get_checkpoint_path(opt)
    start_epoch, start_batch, global_step = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, train_loader, device,
        auto_resume=opt['datasets']['train'].get('auto_resume', True),
        ignore_resume=ignore_resume
    )
    criterion, loss_type = get_loss_criterion(opt, logger)
    train_cfg = opt['train']
    save_every = train_cfg.get('save_checkpoint_epoch', 10)
    try:
        for epoch in range(start_epoch, train_cfg['num_epochs']):
            model.train()
            for batch_idx, (lq, hq) in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue
                lq, hq = lq.to(device), hq.to(device)
                optimizer.zero_grad()
                out = model(lq)
                if isinstance(out, dict):
                    out = out['out']
                metrics = compute_metrics(out, hq, criterion, loss_type)
                metrics['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← добавить
                optimizer.step()
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.log_metrics(
                        metrics['total_loss'].item() if isinstance(metrics['total_loss'], torch.Tensor) else metrics['total_loss'],
                        metrics['psnr'], current_lr, global_step,  # <-- УБРАЛИ .item()
                        model=model, targets=hq, outputs=out, device=device,

## === run_training.py (part 2) ===
                        l1_loss_val=metrics['l1_val'], ffl_loss_val=metrics['ffl_val']
                    )
                global_step += 1
                log_progress(
                    logger=logger,
                    exp_name=opt['name'],
                    epoch=epoch,
                    total_epochs=train_cfg['num_epochs'],
                    batch_idx=batch_idx,
                    total_batches=len(train_loader),
                    loss_val=metrics['total_loss'].item(),
                    psnr_val=metrics['psnr'],  # 🎯 ИСПРАВЛЕНО: передаем чистый float напрямую
                    device=device,
                    print_freq=opt.get('logger', {}).get('print_freq', 10),
                    vram=torch.cuda.memory_allocated(device)/(1024**3) if device.type == 'cuda' else None
                )
            start_batch = 0
            scheduler.step()
            val_freq = train_cfg.get('validation_freq', 1)   # по умолчанию 1
            if (epoch + 1) % val_freq == 0:
                mean_ssim = validator.run_validation(model, epoch, device)
                train_logger.log_validation_metrics(epoch, mean_ssim)
            if (epoch + 1) % save_every == 0:
                save_checkpoint(checkpoint_path, epoch, 0, model, optimizer, scheduler, global_step, is_emergency=False)
                logger.info(f"Чекпоинт сохранён для эпохи {epoch+1}")
    except KeyboardInterrupt:
        logger.warning("Прерывание по Ctrl+C, аварийное сохранение...")
        save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=True)
        train_logger.close()
        sys.exit(0)
if __name__ == '__main__':
    main()

## === libraries/training_checkpoint.py (part 1) ===
import os
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
def get_checkpoint_path(opt):
    """Определяет путь к файлу чекпоинта на основе конфига и аргумента --resume."""
    resume = opt.get('resume')  # из аргументов командной строки (--resume)
    if resume:
        return Path(resume)
    resume_path = opt.get('path', {}).get('resume_path', 'checkpoints/resume.pth')
    return Path(resume_path)
def load_checkpoint(checkpoint_path, model, optimizer, scheduler, train_loader, device, auto_resume, ignore_resume=False):
    start_epoch, start_batch, global_step = 0, 0, 0
    if auto_resume and not ignore_resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model')))
        optimizer.load_state_dict(ckpt.get('optimizer_state_dict', ckpt.get('opt')))
        start_epoch = ckpt.get('epoch', 0)
        is_emergency = ckpt.get('is_emergency', False)
        saved_batch = ckpt.get('batch_idx', 0)
        if is_emergency:
            start_batch = saved_batch + 1
        else:
            start_batch = 0
        if 'scheduler_state_dict' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = start_epoch * len(train_loader) + start_batch
        print(f"Продолжаем с Эпохи: {start_epoch}, Батча: {start_batch} (аварийное: {is_emergency})")
    else:
        print("ℹ️ Чекпоинт не найден или проигнорирован. Старт с чистого листа.")
    return start_epoch, start_batch, global_step
def save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=False):
    save_path = Path(checkpoint_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'is_emergency': is_emergency,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, str(save_path))

## === libraries/training_validator.py (part 1) ===
import os
import sys
import cv2
import torch
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from libraries.logger import get_logger
class VisualValidator:
    def __init__(self, opt):
        self.opt = opt
        self.logger = get_logger()
        log_cfg = self.opt.get('logger', {}).get('include_in_file_log', {})
        self.log_ssim_to_file = log_cfg.get('log_ssim', False)
        self.log_console_cfg = self.opt.get('logger', {}).get('include_in_console', {})
        self.in_channels = opt['network_g']['num_in_ch']
        exp_name = opt.get('name', 'default_experiment')
        dataset_root = opt['path']['dataset_root']
        self.hq_val_dir = os.path.join(dataset_root, "test", "hq_targets")
        self.lq_val_dir = os.path.join(dataset_root, "test", "lq_inputs")
        self.out_val_dir = os.path.join('experiments', exp_name, 'val_predictions')
        os.makedirs(self.out_val_dir, exist_ok=True)
        self.ext = '.jpg' if self.in_channels == 3 else '.tiff'
        train_cfg = opt.get('datasets', {}).get('train', {})
        self.gt_size = train_cfg.get('gt_size', 256)
        self.lq_size = train_cfg.get('lq_size', 128)
    def run_validation(self, model, epoch, device):
        """Прогоняет все тестовые файлы через модель с использованием честного Center Crop"""
        ssim_values = []
        if not os.path.exists(self.lq_val_dir):
            self.logger.warning(f"Папка {self.lq_val_dir} не найдена. Пропускаем.")
            return 0.0
        files = [f for f in os.listdir(self.lq_val_dir) if f.lower().endswith(self.ext)]
        if not files:
            self.logger.warning(f"Нет файлов с расширением {self.ext} в {self.lq_val_dir}")
            return 0.0
        model.eval()
        with torch.no_grad():
            for file_name in files:
                file_path = os.path.join(self.lq_val_dir, file_name)
                base_name = os.path.splitext(file_name)[0].replace('_bayer', '')
                lq = tifffile.imread(file_path).astype(np.float32) / 65535.0
                h_lq, w_lq = lq.shape[:2]
                top_lq = (h_lq - self.lq_size) // 2 if h_lq > self.lq_size else 0
                left_lq = (w_lq - self.lq_size) // 2 if w_lq > self.lq_size else 0
                lq_cropped = lq[top_lq:top_lq+self.lq_size, left_lq:left_lq+self.lq_size, :]

## === libraries/training_validator.py (part 2) ===
                lq_tensor = torch.from_numpy(lq_cropped.transpose(2, 0, 1)).float()
                input_tensor = lq_tensor.unsqueeze(0).to(device)
                output = model(input_tensor)
                if isinstance(output, dict):
                    output = output['out']
                output_np = output.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                final_img = (output_np * 255.0).astype(np.uint8)
                gt_path = os.path.join(self.hq_val_dir, f"{base_name}.png")
                if os.path.exists(gt_path):
                    gt_img = np.array(Image.open(gt_path).convert('RGB'))
                    h_gt, w_gt = gt_img.shape[:2]
                    top_hq = top_lq * 2
                    left_hq = left_lq * 2
                    gt_cropped = gt_img[top_hq:top_hq+self.gt_size, left_hq:left_hq+self.gt_size, :]
                    if gt_cropped.shape[:2] != final_img.shape[:2]:
                        gt_cropped = cv2.resize(gt_cropped, (final_img.shape[1], final_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    ssim_val = ssim(final_img, gt_cropped, channel_axis=2, data_range=255)
                    if self.log_ssim_to_file:
                        self.logger.info(f"[Валидатор] {base_name}: SSIM = {ssim_val:.4f}")
                    ssim_values.append(ssim_val)
                out_name = f"epoch_{epoch}_{base_name}.png"
                out_path = os.path.join(self.out_val_dir, out_name)
                Image.fromarray(final_img).save(out_path)
        model.train()
        mean_ssim = np.mean(ssim_values) if ssim_values else 0.0
        self.logger.info(f"📊 [Валидация] Средний SSIM за эпоху {epoch}: {mean_ssim:.4f}")
        return mean_ssim
    def calculate_ssim(self, restored_path, gt_path):
        restored = np.array(Image.open(restored_path).convert('RGB'))
        gt = np.array(Image.open(gt_path).convert('RGB'))
        return ssim(restored, gt, channel_axis=2, data_range=255)

## === libraries/pipeline_data.py (part 1) ===
import os
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image
import albumentations as A
from skimage.transform import resize
from libraries.logger import get_logger
logger = get_logger()
class RestorationDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.is_train = is_train
        path = config.get('path', {})
        dataset_root = path.get("dataset_root", "datasets/nef_nafnet")
        phase = "train" if is_train else "val"
        self.lq_dir = os.path.join(dataset_root, phase, "lq_inputs")
        self.hq_dir = os.path.join(dataset_root, phase, "hq_targets")
        data_cfg = config.get("datasets", {}).get(phase, {})
        self.gt_size = data_cfg.get("gt_size", 256)
        self.lq_size = data_cfg.get("lq_size", 128)
        self.file_names = sorted(os.listdir(self.lq_dir))
        logger.info(f"Загрузка датасета, {len(self.file_names)} файлов")
        self.use_augment = True
        if self.use_augment and self.is_train:
            self.geom_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            self.geom_transform = None
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        import random
        for _ in range(len(self.file_names)):
            lq_name = self.file_names[idx]
            hq_name = lq_name.replace("_bayer.tiff", ".png")
            lq_path = os.path.join(self.lq_dir, lq_name)
            hq_path = os.path.join(self.hq_dir, hq_name)
            try:
                lq_img = tifffile.imread(lq_path).astype(np.float32) / 65535.0
                hq_img = np.array(Image.open(hq_path).convert('RGB')).astype(np.float32) / 255.0
                if lq_img is None or hq_img is None:
                    raise ValueError
                break

## === libraries/pipeline_data.py (part 2) ===
            except Exception as e:
                logger.warning(f"Ошибка чтения {lq_path}: {e}")
                idx = random.randint(0, len(self.file_names) - 1)
        else:
            raise FileNotFoundError("All files unreadable")
        lq_img = resize(lq_img, (self.lq_size, self.lq_size), preserve_range=True, anti_aliasing=True).astype(lq_img.dtype)
        hq_img = resize(hq_img, (self.gt_size, self.gt_size), preserve_range=True, anti_aliasing=True).astype(hq_img.dtype)
        if self.geom_transform is not None:
            pass
        lq_tensor = torch.from_numpy(lq_img.transpose(2, 0, 1)).float()
        hq_tensor = torch.from_numpy(hq_img.transpose(2, 0, 1)).float()
        return lq_tensor, hq_tensor
def create_restoration_dataset(config, is_train=True):
    return RestorationDataset(config, is_train=is_train)

## === libraries/training_utils.py (part 1) ===
import os
import torch
import shutil
from torch.utils.data import DataLoader
from libraries.config import get_pipeline_config
from libraries.device import get_torch_device
from libraries.pipeline_data import create_restoration_dataset
from libraries.model_utils import create_nafnet_model
from libraries.logger import setup_logger, get_logger
import torch.nn as nn
def log_progress(logger, exp_name, epoch, total_epochs, batch_idx, total_batches,
                 loss_val, psnr_val, device, print_freq, vram=None):
    """Выводит в консоль прогресс обучения с заданной частотой."""
    if batch_idx % print_freq != 0:
        return
    parts = [f"[{exp_name}] Epoch {epoch}/{total_epochs} Batch {batch_idx}/{total_batches}"]
    parts.append(f"Loss: {loss_val:.5f}")
    parts.append(f"PSNR: {psnr_val:.2f} dB")
    if device.type == 'cuda' and vram is not None:
        parts.append(f"VRAM: {vram:.2f}GB")
    logger.info(" | ".join(parts))
def setup_experiment():
    """Загружает конфиг, настраивает логгер, возвращает opt, save_dir, device, logger."""
    opt = get_pipeline_config()
    log_cfg = opt.get('pipeline_logger', {})
    log_file = log_cfg.get('log_file', 'pipeline.log')
    setup_logger(log_file)
    logger = get_logger()
    exp_name = opt.get('name', 'default_exp')
    save_dir = os.path.join('experiments', exp_name)
    device = get_torch_device()
    return opt, save_dir, device, logger
def create_train_loader(opt):
    path_cfg = opt['path']
    train_cfg = opt['datasets']['train']
    dataroot_lq = os.path.join(path_cfg['dataset_root'], 'train', 'lq_inputs')
    dataroot_gt = os.path.join(path_cfg['dataset_root'], 'train', 'hq_targets')
    from libraries.training_dataset_nef import CustomNEFPairDataset
    dataset = CustomNEFPairDataset(dataroot_lq, dataroot_gt, opt=opt)
    return DataLoader(
        dataset,
        batch_size=train_cfg['batch_size_per_gpu'],
        shuffle=True,
        num_workers=train_cfg.get('num_worker_per_gpu', 4),
        pin_memory=(torch.cuda.is_available())
    )
def create_optimizer_and_scheduler(model, opt, total_training_steps=None):
    """
    Создаёт оптимизатор AdamW и планировщик скорости обучения на основе конфига.
    Args:

## === libraries/training_utils.py (part 2) ===
        model: torch.nn.Module
        opt (dict): полный конфигурационный словарь (с секциями 'train' и 'scheduler')
        total_training_steps (int, optional): общее количество итераций (шагов) для OneCycleLR.
            Если не указан, будет вычислен как num_epochs * len(train_loader).
    Returns:
        optimizer, scheduler
    """
    train_cfg = opt['train']
    optim_cfg = train_cfg['optim_g']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg['lr']),
        weight_decay=float(optim_cfg.get('weight_decay', 0.0)),
        betas=optim_cfg.get('betas', (0.9, 0.999))
    )
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', 'CosineAnnealingLR')
    if scheduler_type == 'CosineAnnealingLR':
        t_max = scheduler_cfg.get('T_max', train_cfg['num_epochs'])
        eta_min = float(scheduler_cfg.get('eta_min', 1e-7))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    elif scheduler_type == 'StepLR':
        step_size = scheduler_cfg.get('step_size')
        if step_size is None:
            raise KeyError("StepLR requires 'step_size' in scheduler config")
        gamma = float(scheduler_cfg.get('gamma', 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == 'MultiStepLR':
        milestones = scheduler_cfg.get('milestones')
        if milestones is None:
            raise KeyError("MultiStepLR requires 'milestones' list in scheduler config")
        gamma = float(scheduler_cfg.get('gamma', 0.1))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif scheduler_type == 'ExponentialLR':
        gamma = float(scheduler_cfg.get('gamma', 0.99))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == 'ReduceLROnPlateau':
        mode = scheduler_cfg.get('mode', 'min')
        factor = float(scheduler_cfg.get('factor', 0.1))
        patience = int(scheduler_cfg.get('patience', 10))
        threshold = float(scheduler_cfg.get('threshold', 1e-4))
        cooldown = int(scheduler_cfg.get('cooldown', 0))
        min_lr = float(scheduler_cfg.get('min_lr', 0.0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

## === libraries/training_utils.py (part 3) ===
            optimizer, mode=mode, factor=factor, patience=patience,
            threshold=threshold, cooldown=cooldown, min_lr=min_lr
        )
    elif scheduler_type == 'OneCycleLR':
        if total_training_steps is None:
            raise ValueError("OneCycleLR requires total_training_steps (e.g., num_epochs * len(train_loader))")
        max_lr = float(scheduler_cfg.get('max_lr', optim_cfg['lr']))
        pct_start = float(scheduler_cfg.get('pct_start', 0.3))
        anneal_strategy = scheduler_cfg.get('anneal_strategy', 'cos')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_training_steps,
            pct_start=pct_start, anneal_strategy=anneal_strategy
        )
    elif scheduler_type == 'LambdaLR':
        lr_lambda = scheduler_cfg.get('lr_lambda')
        if lr_lambda is None:
            raise KeyError("LambdaLR requires 'lr_lambda' in scheduler config")
        if isinstance(lr_lambda, str):
            lr_lambda = eval(lr_lambda)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return optimizer, scheduler
def get_loss_criterion(opt, logger):
    loss_type = opt.get('losses', {}).get('type', 'l1')
    if loss_type == 'combined':
        from libraries.training_losses import CombinedLoss
        criterion = CombinedLoss(opt)
        logger.info("Используется комбинированная потеря (L1 + FFL)")
    else:
        criterion = torch.nn.L1Loss()
        logger.info("Используется стандартная L1Loss")
    return criterion, loss_type
def calculate_psnr(mse_loss):
    if mse_loss == 0:
        return float('inf')
    return (20 * torch.log10(1.0 / torch.sqrt(mse_loss))).item()
def compute_metrics(out, target, criterion, loss_type):
    if loss_type == 'combined':
        total_loss, l1_val, ffl_val = criterion(out, target)
    else:
        total_loss = criterion(out, target)
        l1_val = total_loss.item()
        ffl_val = 0.0
    mse = nn.MSELoss()(out, target).detach()
    psnr = calculate_psnr(mse)  # Чистый float
    return {
        'total_loss': total_loss,
        'l1_val': l1_val,
        'ffl_val': ffl_val,

## === libraries/training_utils.py (part 4) ===
        'psnr': psnr  # Безопасно для логгеров, без блокировок CUDA
    }
def cleanup_experiment(opt, save_dir, logger):
    if not opt.get('clean_training', False):
        return False
    logger.info(f"Выборочная очистка файлов обучения в {save_dir} (оставляем debug_visuals и pipeline.log)")
    for subdir in ['checkpoints', 'tb_logs', 'val_predictions']:
        subdir_path = os.path.join(save_dir, subdir)
        if os.path.exists(subdir_path):
            shutil.rmtree(subdir_path)
            logger.info(f"Удалена папка: {subdir_path}")
    for filename in ['train_metrics.csv', 'train_progress.log', 'val_metrics.csv']:
        file_path = os.path.join(save_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Удалён файл: {file_path}")
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'tb_logs'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val_predictions'), exist_ok=True)
    return True

## === libraries/training_logger.py (part 1) ===
import os
import sys
import csv
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from torch.utils.tensorboard import SummaryWriter
from libraries.logger import get_logger
logger = get_logger()
class TrainingLogger:
    def __init__(self, opt, save_dir):
        self.opt = opt
        self.save_dir = save_dir
        self.use_tb = opt.get('logger', {}).get('use_tb_logger', False)
        self.tb_logger = None
        self.prev_psnr = None   # для delta PSNR
        if self.use_tb:
            log_dir = os.path.join(save_dir, 'tb_logs')
            self.tb_logger = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard инициализирован в: {log_dir}")
        else:
            logger.info("TensorBoard отключен в настройках YAML.")
        self.log_csv = opt.get('logger', {}).get('log_csv', False)
        self.log_loss_components = opt.get('logger', {}).get('log_loss_components', True)
        if self.log_csv:
            csv_filename = opt.get('logger', {}).get('csv_file_name', 'train_metrics.csv')
            self.csv_path = os.path.join(save_dir, csv_filename)
            self.csv_headers = [
                'step', 'loss', 'psnr', 'lr',
                'delta_psnr', 'grad_var', 'tv_ratio',
                'vram_alloc_gb', 'vram_res_gb',
                'l1_loss', 'ffl_loss'
            ]
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                logger.info(f"CSV лог создан: {self.csv_path}")
    def log_metrics(self, loss_val, psnr_val, lr_val, global_step,
                    model=None, targets=None, outputs=None, device=None,
                    l1_loss_val=None, ffl_loss_val=None):
        """
        Логирует метрики в TensorBoard, текстовый файл и CSV.
        l1_loss_val, ffl_loss_val - дополнительные компоненты комбинированной потери.
        """
        if self.use_tb and self.tb_logger is not None:
            self.tb_logger.add_scalar('train/loss', loss_val, global_step)
            self.tb_logger.add_scalar('train/psnr', psnr_val, global_step)
            self.tb_logger.add_scalar('train/lr', lr_val, global_step)

## === libraries/training_logger.py (part 2) ===
        logger_opt = self.opt.get('logger', {})
        file_freq = logger_opt.get('file_log_freq', 1)
        delta_psnr = 0.0
        grad_var = 0.0
        tv_ratio = 0.0
        alloc_vram = 0.0
        res_vram = 0.0
        if self.prev_psnr is not None:
            delta_psnr = psnr_val - self.prev_psnr
        self.prev_psnr = psnr_val
        if model is not None:
            grad_norms = [p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None]
            grad_var = np.var(grad_norms) if grad_norms else 0.0
        if targets is not None and outputs is not None:
            tv_out = torch.sum(torch.abs(outputs[:, :, :, :-1] - outputs[:, :, :, 1:])) + \
                     torch.sum(torch.abs(outputs[:, :, :-1, :] - outputs[:, :, 1:, :]))
            tv_tar = torch.sum(torch.abs(targets[:, :, :, :-1] - targets[:, :, :, 1:])) + \
                     torch.sum(torch.abs(targets[:, :, :-1, :] - targets[:, :, 1:, :]))
            tv_ratio = (tv_out / (tv_tar + 1e-8)).item()
        if device is not None and torch.cuda.is_available():
            alloc_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
            res_vram = torch.cuda.memory_reserved(device) / (1024 ** 3)
        if global_step % file_freq == 0:
            log_file_name = logger_opt.get('log_file_name', 'train_progress.log')
            log_path = os.path.join(self.save_dir, log_file_name)
            flags = logger_opt.get('include_in_file_log', {})
            log_parts = [f"[Шаг: {global_step:05d}]"]
            if flags.get('loss', True):
                log_parts.append(f"Loss: {loss_val:.6f}")
            if flags.get('psnr', True):
                log_parts.append(f"PSNR: {psnr_val:.2f} dB")
            if flags.get('learning_rate', True):
                log_parts.append(f"LR: {lr_val:.2e}")
            if flags.get('delta_psnr', True):
                log_parts.append(f"dPSNR: {delta_psnr:+.4f}")
            if flags.get('grad_variance', True):
                log_parts.append(f"GradVar: {grad_var:.2e}")
            if flags.get('tv_ratio', True):
                log_parts.append(f"TV_Ratio: {tv_ratio:.3f}")
            if flags.get('vram_usage', True):
                log_parts.append(f"VRAM: {alloc_vram:.2f}GB/{res_vram:.2f}GB")
            if self.log_loss_components and flags.get('l1_loss', False) and l1_loss_val is not None:
                log_parts.append(f"L1: {l1_loss_val:.6f}")
            if self.log_loss_components and flags.get('ffl_loss', False) and ffl_loss_val is not None:
                log_parts.append(f"FFL: {ffl_loss_val:.6f}")
            log_string = " | ".join(log_parts) + "\n"
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_string)
        if self.log_csv:
            l1 = l1_loss_val if l1_loss_val is not None else 0.0

## === libraries/training_logger.py (part 3) ===
            ffl = ffl_loss_val if ffl_loss_val is not None else 0.0
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step, loss_val, psnr_val, lr_val,
                    delta_psnr, grad_var, tv_ratio, alloc_vram, res_vram,
                    l1, ffl
                ])
    def log_validation_image(self, tag, image_tensor, epoch):
        if self.use_tb and self.tb_logger:
            self.tb_logger.add_image(tag, image_tensor, epoch)
    def close(self):
        if self.tb_logger:
            self.tb_logger.close()
            logger.info("Потоки TensorBoard закрыты.")
    def log_validation_metrics(self, epoch, ssim, psnr_val=None):
        val_csv_path = os.path.join(self.save_dir, 'val_metrics.csv')
        file_exists = os.path.exists(val_csv_path)
        with open(val_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                if psnr_val is not None:
                    writer.writerow(['epoch', 'ssim', 'psnr'])
                else:
                    writer.writerow(['epoch', 'ssim'])
            if psnr_val is not None:
                writer.writerow([epoch, ssim, psnr_val])
            else:
                writer.writerow([epoch, ssim])
        logger.info(f"Валидационные метрики сохранены в {val_csv_path}")

## === libraries/pipeline_generation_core.py (part 1) ===
import sys
import cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from libraries.degradation_ops import (
    create_psf_kernel, add_uncorrelated_noise_float, add_correlated_noise_float,
    apply_bayer_mask_float, extract_bayer_subchannels
)
def generate_lq_from_hq(hq_rgb, config):
    """
    Генерирует пару (lq_packed, hq_target) для обучения.
    lq_packed: 4-канальный RGGB (H_small, W_small, 4)  uint16
    hq_target: резкое RGB (H_original, W_original, 3) uint8
    """
    proc_cfg = config.get('process_data', {})
    downscale = proc_cfg.get('downscale_factor', 1)
    noise_cfg = proc_cfg.get('noise', {})
    add_noise = noise_cfg.get('add', False)
    snr_db = noise_cfg.get('snr_db', 30)
    correlated = noise_cfg.get('correlated', False)
    psf_sigma = noise_cfg.get('psf_sigma', 1.5)
    hq_target = np.clip(hq_rgb, 0, 255).astype(np.uint8)   # (H, W, 3)
    if downscale > 1:
        h, w = hq_rgb.shape[:2]
        new_h, new_w = h // downscale, w // downscale
        hq_small = cv2.resize(hq_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        hq_small = hq_rgb.copy()
    hq_small_float = hq_small.astype(np.float32) / 255.0
    psf_kernel = create_psf_kernel(psf_sigma)
    hq_blurred_float = np.zeros_like(hq_small_float)
    for c in range(3):
        hq_blurred_float[..., c] = cv2.filter2D(hq_small_float[..., c], -1, psf_kernel)
    bayer_float = apply_bayer_mask_float(hq_blurred_float, pattern='RGGB')
    if add_noise:
        if correlated:
            bayer_float = add_correlated_noise_float(bayer_float, snr_db, psf_kernel)
        else:
            bayer_float = add_uncorrelated_noise_float(bayer_float, snr_db)
    lq_packed = extract_bayer_subchannels(bayer_float)   # (H_small/2, W_small/2, 4)
    return lq_packed, hq_target

## === libraries/pipeline_prepare.py (part 1) ===
"""
Модуль подготовки датасета: проверка, очистка, генерация через прямые вызовы.
"""
import os
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from libraries.logger import get_logger
from libraries.pipeline_generation_io import process_source_images
logger = get_logger()
def clean_dataset_folders(dataset_root):
    """Удаляет папки train и test внутри dataset_root."""
    train_path = os.path.join(dataset_root, "train")
    test_path = os.path.join(dataset_root, "test")
    for p in [train_path, test_path]:
        if os.path.exists(p):
            shutil.rmtree(p)
            logger.info(f"Очистка: удалена папка {p}")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    logger.info(f"Очистка датасета завершена, корень {dataset_root} сохранён")
def is_dataset_empty(dataset_root):
    """Проверяет, пуст ли датасет (отсутствуют lq_inputs в train)."""
    lq_dir = os.path.join(dataset_root, "train", "lq_inputs")
    if not os.path.exists(lq_dir):
        return True
    return len(os.listdir(lq_dir)) == 0
def ensure_dataset_ready(config, clean_dataset=False):
    """
    Гарантирует, что датасет готов к использованию.
    Если clean_dataset=True – сначала очищает train/test.
    Если датасет пуст – запускает генерацию напрямую.
    """
    dataset_root = config.get("dataset_root", "datasets/nef_nafnet")
    logger.info(f"Работа с датасетом, корень: {dataset_root}")
    if clean_dataset:
        logger.info("Флаг clean_dataset=True: очищаем папки train/test")
        clean_dataset_folders(dataset_root)
    if is_dataset_empty(dataset_root):
        logger.info("Датасет пуст или не найден. Запуск генерации...")
        config['clean_generation'] = clean_dataset
        process_source_images(config)
        logger.info("Генерация датасета завершена")
    else:
        lq_dir = os.path.join(dataset_root, "train", "lq_inputs")
        files_count = len(os.listdir(lq_dir))
        logger.info(f"Датасет уже существует ({files_count} патчей). Генерация пропущена.")

## === libraries/device.py (part 1) ===
"""
Модуль определения вычислительного устройства (CPU/CUDA).
"""
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
def get_torch_device():
    """
    Определяет доступное устройство: CUDA (GPU) или CPU.
    Возвращает объект torch.device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔧 [Device] Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("🔧 [Device] CUDA не найдена, используется CPU")
    return device
def move_model_to_device(model, device):
    """Переносит модель на указанное устройство."""
    return model.to(device)

## === libraries/pipeline_smoke.py (part 1) ===
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from libraries.logger import get_logger
logger = get_logger()
try:
    from libraries.training_losses import FocalFrequencyLoss
    HAS_FFL = True
except ModuleNotFoundError:
    logger.warning("FocalFrequencyLoss не найден! Используется только L1-Loss.")
    HAS_FFL = False
def run_smoke_test(model, dataset, config, device):
    """
    Нагрузочное тестирование градиентных потоков и видеопамяти GPU.
    Проверяет связку: NAFNet + FocalFrequencyLoss + AdamW.
    """
    logger.info("=== Запуск модуля стресс-тестирования (Smoke Test) ===")
    logger.info(f"Вычислительное устройство: {device}")
    model = model.to(device)
    model.train()
    l1_loss_fn = nn.L1Loss()
    if HAS_FFL:
        ffl_cfg = config.get("losses", {}).get("ffl_opt", {})
        ffl_alpha = ffl_cfg.get("alpha", 1.0)
        ffl_weight = ffl_cfg.get("loss_weight", 1.0)
        ffl_loss_fn = FocalFrequencyLoss(loss_weight=ffl_weight, alpha=ffl_alpha)
        logger.info(f"Инициализирован FocalFrequencyLoss (weight={ffl_weight})")
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    try:
        lq_tensor, hq_tensor = dataset[0] # Берем первый сэмпл датасета
        lq_batch = lq_tensor.unsqueeze(0).to(device)
        hq_batch = hq_tensor.unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Не удалось подготовить тестовый батч данных: {e}")
        raise e
    try:
        optimizer.zero_grad()
        pred_batch = model(lq_batch)
        logger.info(f"Прямой проход успешен. Размерность выхода: {list(pred_batch.shape)}")
        if hq_batch.shape[2:] != pred_batch.shape[2:]:
            logger.warning(f"⚠️ Масштаб таргета {list(hq_batch.shape)} не совпадает с предсказанием {list(pred_batch.shape)}. Интерполируем.")
            hq_batch = F.interpolate(hq_batch, size=(pred_batch.shape[2], pred_batch.shape[3]), mode='bilinear', align_corners=False)
        loss = l1_loss_fn(pred_batch, hq_batch)
        if HAS_FFL:
            loss += ffl_loss_fn(pred_batch, hq_batch)

## === libraries/pipeline_smoke.py (part 2) ===
        logger.info(f"Расчет лосса успешен. Значение: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        logger.info("Обратный проход градиентов и шаг AdamW выполнены без ошибок.")
        logger.info("🚀 Стендовый нагрузочный тест (Smoke Test) успешно пройден!")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Out of Memory на GPU!")
        else:
            logger.error(f"Сбой во время вычислений на графе нейросети: {e}")
        raise e

## === libraries/config.py (part 1) ===
import os
import sys
import argparse
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from libraries.logger import get_logger
logger = get_logger()
RED = "\033[91m"
RESET = "\033[0m"
CROSS = f"{RED}✘{RESET}"
def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Tanahen Image Restoration Pipeline")
    parser.add_argument("-opt", type=str, required=True, help="Путь к конфигурационному файлу YAML")
    parser.add_argument("--clean", action="store_true", help="[DEPRECATED] Очистить всё (датасет + визуализации)")
    parser.add_argument("--clean-dataset", action="store_true", help="Очистить папки train/test датасета")
    parser.add_argument("--clean-visuals", action="store_true", help="Очистить папку с визуализациями")
    parser.add_argument("--clean-training", action="store_true", help="Очистить эксперимент (удалить папку experiments/имя)")
    parser.add_argument("--resume", type=str, default=None, help="Путь к чекпоинту для продолжения обучения")
    return parser.parse_args()
def load_yaml_config(config_path):
    """Чтение и валидация YAML-конфигурации."""
    if not os.path.exists(config_path):
        logger.error(f"{CROSS}\tКонфигурационный файл не найден: {config_path}")
        sys.exit(1)  # Завершает работу с кодом ошибки
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Конфигурация успешно загружена из файла: {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Ошибка синтаксиса в YAML-файле: {e}")
            raise e
def get_pipeline_config():
    args = parse_args()
    config = load_yaml_config(args.opt)
    if args.clean:
        args.clean_dataset = True
        args.clean_visuals = True
    config["clean_dataset"] = args.clean_dataset
    config["clean_visuals"] = args.clean_visuals
    config["clean_training"] = args.clean_training
    config["resume"] = args.resume
    config["opt_path"] = args.opt
    exp_name = config.get('name')
    if exp_name:
        vis_cfg = config.get('visuals_logger', {})
        if 'output_dir' in vis_cfg:
            vis_cfg['output_dir'] = vis_cfg['output_dir'].replace('{name}', exp_name)

## === libraries/config.py (part 2) ===
        log_cfg = config.get('pipeline_logger', {})
        if 'log_file' in log_cfg:
            log_cfg['log_file'] = log_cfg['log_file'].replace('{name}', exp_name)
    resume_cfg = config.get('path', {})
    if 'resume_path' in resume_cfg:
        resume_cfg['resume_path'] = resume_cfg['resume_path'].replace('{name}', exp_name)
    return config

## === libraries/training_dataset_nef.py (part 1) ===
import os
import sys
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torchvision.transforms.functional as TF
class CustomNEFPairDataset(Dataset):
    def __init__(self, dataroot_lq, dataroot_gt, opt=None):
        self.dataroot_lq = dataroot_lq
        self.dataroot_gt = dataroot_gt
        self.filenames = [
            os.path.splitext(f)[0].replace('_bayer', '')
            for f in os.listdir(dataroot_lq)
            if f.endswith('.tiff')
        ]
        train_cfg = opt.get('datasets', {}).get('train', {}) if opt else {}
        self.upscale_factor = opt.get('datasets', {}).get('train', {}).get('upscale_factor', 2)
        self.gt_size = train_cfg.get('gt_size', 256)   # размер HQ (должен быть 2 * lq_size)
        self.lq_size = train_cfg.get('lq_size', 128)
        self.use_flip = train_cfg.get('use_flip', False)
        self.use_rot = train_cfg.get('use_rot', False)
        assert self.gt_size == self.upscale_factor * self.lq_size, \
            f"gt_size ({self.gt_size}) must be {self.upscale_factor} * lq_size ({self.lq_size})"
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        name = self.filenames[idx]
        lq_path = os.path.join(self.dataroot_lq, f"{name}_bayer.tiff")
        lq_packed = tifffile.imread(lq_path).astype(np.float32) / 65535.0   # (H_lq, W_lq, 4)
        gt_path = os.path.join(self.dataroot_gt, f"{name}.png")
        gt_img = Image.open(gt_path).convert('RGB')
        gt_rgb = np.array(gt_img).astype(np.float32) / 255.0                # (H_hq, W_hq, 3)
        h_lq, w_lq = lq_packed.shape[:2]
        h_hq, w_hq = gt_rgb.shape[:2]
        target_h = self.upscale_factor * h_lq
        target_w = self.upscale_factor * w_lq
        if h_hq != target_h or w_hq != target_w:
            from skimage.transform import resize
            gt_rgb = resize(gt_rgb, (target_h, target_w), preserve_range=True)
            h_hq, w_hq = target_h, target_w        
        top_lq = random.randint(0, h_lq - self.lq_size) if h_lq > self.lq_size else 0
        left_lq = random.randint(0, w_lq - self.lq_size) if w_lq > self.lq_size else 0
        lq_cropped = lq_packed[top_lq:top_lq+self.lq_size, left_lq:left_lq+self.lq_size, :]
        top_hq = top_lq * 2
        left_hq = left_lq * 2

## === libraries/training_dataset_nef.py (part 2) ===
        gt_cropped = gt_rgb[top_hq:top_hq+self.gt_size, left_hq:left_hq+self.gt_size, :]
        lq_tensor = torch.from_numpy(lq_cropped.transpose(2, 0, 1)).float()   # (4, lq_size, lq_size)
        gt_tensor = torch.from_numpy(gt_cropped.transpose(2, 0, 1)).float()   # (3, gt_size, gt_size)
        if self.use_flip:
            if random.random() > 0.5:
                lq_tensor = TF.hflip(lq_tensor)
                gt_tensor = TF.hflip(gt_tensor)
            if random.random() > 0.5:
                lq_tensor = TF.vflip(lq_tensor)
                gt_tensor = TF.vflip(gt_tensor)
        if self.use_rot:
            k = random.randint(0, 3)   # 0 – без поворота, 1 – 90°, 2 – 180°, 3 – 270°
            if k > 0:
                lq_tensor = TF.rotate(lq_tensor, k * 90)
                gt_tensor = TF.rotate(gt_tensor, k * 90)
        return lq_tensor, gt_tensor

## === libraries/model_utils.py (part 1) ===
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from basicsr.models.archs.NAFNet_arch import NAFNet
from libraries.logger import get_logger
logger = get_logger()
class NAFNetDemosaicSuperResolutionWrapper(nn.Module):
    """
    Академическая обёртка модели для честной демозаики.
    Принимает 4-канальный subchannels-пакет (H/2, W/2, 4) и с помощью 
    субпиксельного сдвига (PixelShuffle) перестраивает его в RGB (H, W, 3).
    """
    def __init__(self, original_model, in_channels=4, out_channels=3, upscale_factor=2):
        super().__init__()
        self.net = original_model
        mid_channels = out_channels * (upscale_factor ** 2)
        self.upsample_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )        
        self.upsample_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
    def forward(self, x):
        out = self.net(x)
        return self.upsample_block(out)
def create_nafnet_model(config, device, pretrained_path=None):
    """
    Создаёт модель NAFNet.
    Если указан pretrained_path, загружает предобученные веса,
    заменяя conv_first для работы с 4 входными каналами.
    """
    net_cfg = config.get("network_g", {})
    in_ch = net_cfg.get("num_in_ch", 4)   # 4 входных канала (RGGB)
    out_ch = net_cfg.get("num_out_ch", 3)  # 3 выходных канала (RGB)
    width = net_cfg.get("width", 32)
    middle_blk_num = net_cfg.get("middle_blk_num", 12)
    enc_blk_nums = net_cfg.get("enc_blk_nums", [2, 2, 4, 8])
    dec_blk_nums = net_cfg.get("dec_blk_nums", [2, 2, 2, 2])
    raw_model = NAFNet(
        img_channel=in_ch,  # Уже стоит 4
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums
    )

## === libraries/model_utils.py (part 2) ===
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Загрузка предобученных весов из {pretrained_path}")
        pretrained_weights = torch.load(pretrained_path, map_location='cpu')
        pretrained_dict = pretrained_weights
        if 'params' in pretrained_weights:
            pretrained_dict = pretrained_weights['params']
        elif 'state_dict' in pretrained_weights:
            pretrained_dict = pretrained_weights['state_dict']
        model_dict = raw_model.state_dict()
        pretrained_dict_filtered = {}
        skipped_layers = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict_filtered[k] = v
                else:
                    skipped_layers.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
            else:
                skipped_layers.append(f"{k} (not in model)")
        if skipped_layers:
            logger.warning(f"Пропущенные слои из-за несовпадения размерностей: {', '.join(skipped_layers)}")
        model_dict.update(pretrained_dict_filtered)
        raw_model.load_state_dict(model_dict, strict=False)  # strict=False позволяет пропустить отсутствующие ключи
        logger.info(f"Загружено {len(pretrained_dict_filtered)} слоёв из предобученной модели.")
    else:
        logger.info("Предобученные веса не указаны или не найдены. Обучение с нуля.")
    upscale_factor = net_cfg.get("upscale_factor", 2)
    model = NAFNetDemosaicSuperResolutionWrapper(raw_model, in_channels=in_ch, out_channels=out_ch, upscale_factor=upscale_factor)    
    model = model.to(device)
    logger.info(f"🚀 [Архитектура] Честный демозаик NAFNet создан!")
    logger.info(f"Вход (RAW subchannels): [{in_ch} ch] -> Выход (PixelShuffle Апскейл): [{out_ch} ch RGB]")
    return model

## === libraries/evaluate_baseline.py (part 1) ===
"""
Модуль сравнения с классическими методами демозаики (билинейная, MHC).
Реализует полноценный алгоритм Malvar-He-Cutler (MHC) для паттерна RGGB.
"""
import os
import sys
import numpy as np
import tifffile
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from scipy.ndimage import zoom, convolve as conv2d
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from libraries.logger import get_logger
logger = get_logger()
def bilinear_demosaic(lq_4ch):
    """
    Билинейная интерполяция каждого из 4 каналов (R, G1, G2, B) с последующим
    усреднением зелёных и сборкой RGB.
    lq_4ch: (H, W, 4) массив float32 в диапазоне [0,1].
    Возвращает: (2H, 2W, 3) массив float32.
    """
    h, w = lq_4ch.shape[:2]
    r_up = zoom(lq_4ch[:,:,0], 2, order=1)
    g1_up = zoom(lq_4ch[:,:,1], 2, order=1)
    g2_up = zoom(lq_4ch[:,:,2], 2, order=1)
    b_up = zoom(lq_4ch[:,:,3], 2, order=1)
    g_up = (g1_up + g2_up) / 2.0
    rgb = np.stack([r_up, g_up, b_up], axis=-1)
    return np.clip(rgb, 0.0, 1.0)
def mhc_demosaic(lq_4ch):
    """
    Математически точная реализация градиентной демозаики Malvar-He-Cutler (MHC)
    с коррекцией межканальных разностей яркости для паттерна RGGB.
    Вход: (H, W, 4) каналы R, G1, G2, B (каждый размером H x W).
    Выход: (2H, 2W, 3) RGB в диапазоне [0,1].
    """
    h, w = lq_4ch.shape[:2]
    bayer = np.zeros((2*h, 2*w), dtype=np.float32)
    bayer[0::2, 0::2] = lq_4ch[:, :, 0] # R
    bayer[0::2, 1::2] = lq_4ch[:, :, 1] # G1
    bayer[1::2, 0::2] = lq_4ch[:, :, 2] # G2
    bayer[1::2, 1::2] = lq_4ch[:, :, 3] # B
    out_r = np.zeros_like(bayer)
    out_g = np.zeros_like(bayer)
    out_b = np.zeros_like(bayer)
    out_r[0::2, 0::2] = bayer[0::2, 0::2]
    out_g[0::2, 1::2] = bayer[0::2, 1::2]
    out_g[1::2, 0::2] = bayer[1::2, 0::2]

## === libraries/evaluate_baseline.py (part 2) ===
    out_b[1::2, 1::2] = bayer[1::2, 1::2]
    kernel_G_at_R_B = np.array([
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  2,  0,  0],
        [-1,  2,  4,  2, -1],
        [ 0,  0,  2,  0,  0],
        [ 0,  0, -1,  0,  0]
    ], dtype=np.float32) / 8.0
    g_interp = conv2d(bayer, kernel_G_at_R_B, mode='mirror')
    out_g = np.where(out_g > 0, out_g, g_interp)
    out_g[0::2, 1::2] = bayer[0::2, 1::2]
    out_g[1::2, 0::2] = bayer[1::2, 0::2]
    kernel_R_B_at_G_row = np.array([
        [ 0,  0,  0.5,  0,  0],
        [ 0, -1,  0,   -1,  0],
        [-1,  4,  5,    4, -1],
        [ 0, -1,  0,   -1,  0],
        [ 0,  0,  0.5,  0,  0]
    ], dtype=np.float32) / 8.0
    out_r_bilin = zoom(lq_4ch[:,:,0], 2, order=1)
    out_b_bilin = zoom(lq_4ch[:,:,3], 2, order=1)
    out_r = out_g + (out_r_bilin - out_g)
    out_b = out_g + (out_b_bilin - out_g)
    out_r[0::2, 0::2] = bayer[0::2, 0::2]
    out_b[1::2, 1::2] = bayer[1::2, 1::2]
    rgb = np.stack([out_r, out_g, out_b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)    
def compute_metrics(pred, gt, data_range=1.0):
    """Вычисляет PSNR и SSIM для двух изображений в диапазоне [0, data_range]."""
    psnr = peak_signal_noise_ratio(gt, pred, data_range=data_range)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=data_range)
    return psnr, ssim
def run_baseline_evaluation(config):
    """
    Загружает тестовые данные из dataset_root, применяет билинейную и MHC демозаику,
    логирует средние PSNR и SSIM.
    """
    if 'path' in config and 'dataset_root' in config['path']:
        dataset_root = config['path']['dataset_root']
    else:
        dataset_root = config.get('dataset_root', 'datasets/nef_nafnet')
    test_lq_dir = os.path.join(dataset_root, 'test', 'lq_inputs')
    test_gt_dir = os.path.join(dataset_root, 'test', 'hq_targets')    
    if not os.path.exists(test_lq_dir) or not os.path.exists(test_gt_dir):
        logger.error(f"Тестовые папки не найдены: {test_lq_dir} или {test_gt_dir}")
        return
    files = [f for f in os.listdir(test_lq_dir) if f.endswith('_bayer.tiff')]
    if not files:
        logger.warning("Нет тестовых файлов в %s", test_lq_dir)
        return

## === libraries/evaluate_baseline.py (part 3) ===
    psnr_bilin, ssim_bilin = [], []
    psnr_mhc, ssim_mhc = [], []
    for fname in files:
        base = fname.replace('_bayer.tiff', '')
        lq_path = os.path.join(test_lq_dir, fname)
        gt_path = os.path.join(test_gt_dir, f"{base}.png")
        if not os.path.exists(gt_path):
            logger.warning(f"Нет эталона для {base}, пропускаем")
            continue
        lq = tifffile.imread(lq_path).astype(np.float32) / 65535.0   # (H, W, 4)
        gt = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0
        rgb_bilin = bilinear_demosaic(lq)
        gt_resized = resize(gt, rgb_bilin.shape[:2], preserve_range=True)
        psnr_b, ssim_b = compute_metrics(rgb_bilin, gt_resized)
        psnr_bilin.append(psnr_b)
        ssim_bilin.append(ssim_b)
        rgb_mhc = mhc_demosaic(lq)
        if rgb_mhc.shape != rgb_bilin.shape:
            rgb_mhc = resize(rgb_mhc, rgb_bilin.shape[:2], preserve_range=True)
        psnr_m, ssim_m = compute_metrics(rgb_mhc, gt_resized)
        psnr_mhc.append(psnr_m)
        ssim_mhc.append(ssim_m)
        logger.info(f"{base}: Bilinear PSNR={psnr_b:.2f} SSIM={ssim_b:.4f} | MHC PSNR={psnr_m:.2f} SSIM={ssim_m:.4f}")
    if psnr_bilin:
        logger.info("=== BASELINE SUMMARY ===")
        logger.info(f"Bilinear  -> Mean PSNR = {np.mean(psnr_bilin):.2f} dB, Mean SSIM = {np.mean(ssim_bilin):.4f}")
        logger.info(f"MHC       -> Mean PSNR = {np.mean(psnr_mhc):.2f} dB, Mean SSIM = {np.mean(ssim_mhc):.4f}")
    else:
        logger.error("Не удалось обработать ни одного файла")

## === libraries/pipeline_generation_io.py (part 1) ===
import os
import sys
import random
import numpy as np
import imageio.v3 as iio
import tifffile
from PIL import Image
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from libraries.pipeline_generation_core import generate_lq_from_hq
from libraries.logger import get_logger
logger = get_logger()
def load_rgb(filepath):
    try:
        img = iio.imread(filepath)
        if img.dtype != np.uint8:
            img = (img / 65535.0 * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    except Exception as e:
        logger.warning(f"Ошибка чтения {filepath}: {e}")
        return None
def process_source_images(config):
    path_opt = config.get('path', {})
    source_dir = os.path.expanduser(path_opt.get('source_images_dir', 'source_images'))
    dataset_root = path_opt.get('dataset_root', 'datasets/nef_nafnet')
    train_ratio = config.get('train_ratio', 0.95)
    seed = config.get('manual_seed', 42)
    clean = config.get('clean_generation', False)
    train_hq = os.path.join(dataset_root, 'train', 'hq_targets')
    train_lq = os.path.join(dataset_root, 'train', 'lq_inputs')
    test_hq = os.path.join(dataset_root, 'test', 'hq_targets')
    test_lq = os.path.join(dataset_root, 'test', 'lq_inputs')
    if clean:
        import shutil
        logger.info("Очистка датасета...")
        for p in [train_hq, train_lq, test_hq, test_lq]:
            if os.path.exists(p):
                shutil.rmtree(p)
    for p in [train_hq, train_lq, test_hq, test_lq]:
        os.makedirs(p, exist_ok=True)
    extensions = ('.nef', '.cr2', '.dng', '.arw', '.jpg', '.jpeg', '.png', '.tiff', '.tif')
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(extensions)]
    if not files:
        logger.error(f"Нет файлов в {source_dir}")
        return

## === libraries/pipeline_generation_io.py (part 2) ===
    random.seed(seed)
    random.shuffle(files)
    split = int(len(files) * train_ratio)
    train_files, test_files = files[:split], files[split:]
    def process_file_list(file_list, subset):
        for fname in tqdm(file_list, desc=f"Генерация {subset}"):
            hq = load_rgb(os.path.join(source_dir, fname))
            if hq is None:
                continue
            lq_packed, hq_target = generate_lq_from_hq(hq, config)
            base = os.path.splitext(fname)[0]
            hq_dir = train_hq if subset == 'train' else test_hq
            lq_dir = train_lq if subset == 'train' else test_lq
            Image.fromarray(hq_target).save(os.path.join(hq_dir, f"{base}.png"))
            lq_16bit = np.clip(lq_packed * 65535.0, 0, 65535).astype(np.uint16)
            tifffile.imwrite(os.path.join(lq_dir, f"{base}_bayer.tiff"), lq_16bit, photometric='minisblack')            
    process_file_list(train_files, 'train')
    process_file_list(test_files, 'test')
    logger.info("Генерация завершена.")

## === libraries/pipeline_dataset.py (part 1) ===
"""
Модуль управления датасетом: проверка, очистка, генерация через prepare_dataset.py.
"""
import os
import subprocess
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from libraries.logger import get_logger
logger = get_logger()
def clean_dataset_folders(dataset_root):
    """Удаляет папки train и test внутри dataset_root, оставляя корень."""
    train_path = os.path.join(dataset_root, "train")
    test_path = os.path.join(dataset_root, "test")
    for p in [train_path, test_path]:
        if os.path.exists(p):
            shutil.rmtree(p)
            logger.info(f"Очистка: удалена папка {p}")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    logger.info(f"Очистка датасета завершена, корень {dataset_root} сохранён")
def is_dataset_empty(dataset_root):
    """Проверяет, пуст ли датасет (отсутствуют lq_inputs в train)."""
    lq_dir = os.path.join(dataset_root, "train", "lq_inputs")
    if not os.path.exists(lq_dir):
        return True
    return len(os.listdir(lq_dir)) == 0
def ensure_dataset_ready(config, opt_path, clean_dataset=False):
    """
    Гарантирует, что датасет готов к использованию.
    Если clean_dataset=True – сначала очищает train/test.
    Если датасет пуст – запускает prepare_dataset.py.
    """
    path = config.get('path', None)
    if path is None:
        sys.exit(0)
    dataset_root = path.get("dataset_root", "datasets/nef_nafnet")
    logger.info(f"Работа с датасетом, корень: {dataset_root}")
    if clean_dataset:
        logger.info("Флаг clean_dataset=True: очищаем папки train/test")
        clean_dataset_folders(dataset_root)
    if is_dataset_empty(dataset_root):
        logger.info("Датасет пуст или не найден. Запуск prepare_dataset.py...")
        cmd = [sys.executable, "prepare_dataset.py", "-opt", opt_path]
        try:
            subprocess.run(cmd, check=True)
            logger.info("prepare_dataset.py выполнен успешно")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при запуске prepare_dataset.py: {e}")

## === libraries/pipeline_dataset.py (part 2) ===
            raise
    else:
        lq_dir = os.path.join(dataset_root, "train", "lq_inputs")
        files_count = len(os.listdir(lq_dir))
        logger.info(f"Датасет уже существует ({files_count} патчей). Генерация пропущена.")
def create_restoration_dataset(config, is_train=True):
    """Создаёт и возвращает экземпляр RestorationDataset."""
    from libraries.pipeline_data import RestorationDataset
    return RestorationDataset(config, is_train=is_train)

## === libraries/pipeline_visuals.py (part 1) ===
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from libraries.logger import get_logger
logger = get_logger()
def bilinear_demosaic_rggb(lq_4ch):
    """
    Быстрая билинейная демозаика 4-канального RGGB.
    Вход: numpy array формы (H, W, 4) в диапазоне [0,1]
    Выход: RGB изображение формы (2H, 2W, 3) в диапазоне [0,1]
    """
    from scipy.ndimage import zoom
    r = lq_4ch[:,:,0]
    g1 = lq_4ch[:,:,1]
    g2 = lq_4ch[:,:,2]
    b = lq_4ch[:,:,3]
    r_up = zoom(r, 2, order=1)
    g1_up = zoom(g1, 2, order=1)
    g2_up = zoom(g2, 2, order=1)
    b_up = zoom(b, 2, order=1)
    g_up = (g1_up + g2_up) / 2.0
    rgb = np.stack([r_up, g_up, b_up], axis=-1)
    return np.clip(rgb, 0.0, 1.0)
def clean_old_visuals(output_dir, filenames):
    """Удаление старых отрендеренных превью из целевой директории."""
    for fname in filenames:
        full_path = os.path.join(output_dir, fname)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                logger.info(f"Старый файл визуализации удален: {full_path}")
            except Exception as e:
                logger.warning(f"Не удалось удалить {full_path}: {e}")
def save_tensor_as_image(tensor, output_path):
    """
    Конвертирует PyTorch тензор в изображение и сохраняет.
    Поддерживает: 3-канальный RGB (C, H, W) и 4-канальный RGGB (C, H, W).
    Для 4-канального применяет билинейную демозаику.
    """
    img_np = tensor.detach().cpu().numpy()
    if img_np.ndim == 3:
        if img_np.shape[0] == 4:
            img_np = img_np.transpose(1, 2, 0)   # (H, W, 4)
            img_rgb = bilinear_demosaic_rggb(img_np)  # (2H, 2W, 3) float [0,1]
            img_np = (img_rgb * 255.0).astype(np.uint8)
        else:

## === libraries/pipeline_visuals.py (part 2) ===
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    else:
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, img_np)
    logger.info(f"Диагностическое превью сохранено в: {output_path}")
def run_visual_control(dataset, config):
    """Основная функция визуального верификационного контроля датасета."""
    logger.info("=== Запуск модуля визуального контроля данных ===")
    vis_cfg = config.get("visuals_logger", {})
    output_dir = vis_cfg.get("output_dir", "samples/debug_visuals")
    lq_name = vis_cfg.get("lq_preview_name", "debug_lq_preview.png")
    gt_name = vis_cfg.get("gt_reference_name", "debug_gt_reference.png")
    os.makedirs(output_dir, exist_ok=True)
    if config.get("clean_visuals", False):
        clean_old_visuals(output_dir, [lq_name, gt_name])
    if len(dataset) == 0:
        logger.error("Ошибка контроля: Датасет пуст!")
        raise IndexError("Dataset is empty")
    test_idx = np.random.randint(0, len(dataset))
    lq_tensor, hq_tensor = dataset[test_idx]
    logger.info(f"Контрольный сэмпл #{test_idx} успешно извлечен из выборки.")
    lq_path = os.path.join(output_dir, lq_name)
    gt_path = os.path.join(output_dir, gt_name)
    save_tensor_as_image(lq_tensor, lq_path)
    save_tensor_as_image(hq_tensor, gt_path)
    logger.info("📸 Визуальный тест успешно завершен.")

## === libraries/logger.py (part 1) ===
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
def setup_logger(log_file="pipeline.log"):
    """
    Инициализирует двухпоточную систему логирования.
    При повторном вызове перенастраивает вывод в новый файл.
    """
    logger = logging.getLogger("TanahenPipeline")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    log_format = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    import os
    log_dir = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger
def get_logger():
    """Возвращает инициализированный логгер для вызовов внутри модулей."""
    return logging.getLogger("TanahenPipeline")

## === libraries/degradation_ops.py (part 1) ===
import sys
import numpy as np
import cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
def create_psf_kernel(sigma, size=15):
    ax = np.linspace(-(size//2), size//2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)
def add_correlated_noise(img, snr_db, psf_kernel):
    signal_power = np.mean(img ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    white_noise = np.random.normal(0, np.sqrt(noise_power), img.shape)
    correlated_noise = cv2.filter2D(white_noise, -1, psf_kernel)
    return np.clip(img + correlated_noise, 0, None)
def add_uncorrelated_noise(img, snr_db):
    signal_power = np.mean(img ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), img.shape)
    return np.clip(img + noise, 0, None)
def apply_bayer_mask(rgb, pattern='RGGB'):
    h, w, _ = rgb.shape
    bayer = np.zeros((h, w), dtype=rgb.dtype)
    bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]   # R
    bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]   # G1
    bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]   # G2
    bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]   # B
    return bayer
def extract_bayer_subchannels(bayer_2d):
    h, w = bayer_2d.shape
    h = h - (h % 2)
    w = w - (w % 2)
    bayer_2d = bayer_2d[:h, :w]
    r = bayer_2d[0::2, 0::2]
    g1 = bayer_2d[0::2, 1::2]
    g2 = bayer_2d[1::2, 0::2]
    b = bayer_2d[1::2, 1::2]
    return np.stack([r, g1, g2, b], axis=2)
def add_uncorrelated_noise_float(img_float, snr_db):
    """
    Добавляет белый гауссовский шум к изображению в формате float [0,1].
    """
    signal_power = np.mean(img_float ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), img_float.shape)
    return np.clip(img_float + noise, 0.0, 1.0)
def add_correlated_noise_float(img_float, snr_db, psf_kernel):
    """
    Добавляет коррелированный (свёрнутый с PSF) шум к изображению float [0,1].

## === libraries/degradation_ops.py (part 2) ===
    """
    signal_power = np.mean(img_float ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    white_noise = np.random.normal(0, np.sqrt(noise_power), img_float.shape)
    correlated_noise = cv2.filter2D(white_noise, -1, psf_kernel)
    return np.clip(img_float + correlated_noise, 0.0, 1.0)
def apply_bayer_mask_float(rgb_float, pattern='RGGB'):
    """
    Преобразует полноцветное RGB float [0,1] в одноканальный Bayer массив float [0,1].
    """
    h, w, _ = rgb_float.shape
    bayer = np.zeros((h, w), dtype=np.float32)
    bayer[0::2, 0::2] = rgb_float[0::2, 0::2, 0]   # R
    bayer[0::2, 1::2] = rgb_float[0::2, 1::2, 1]   # G1
    bayer[1::2, 0::2] = rgb_float[1::2, 0::2, 1]   # G2
    bayer[1::2, 1::2] = rgb_float[1::2, 1::2, 2]   # B
    return bayer

## === libraries/training_losses.py (part 1) ===
import sys
import torch
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
class FocalFrequencyLoss(nn.Module):
    """
    Частотная функция потерь (п. 9 ТЗ).
    Оптимизирует разность спектров предсказания и таргета в частотной области 2D FFT.
    Помогает NAFNet безошибочно достраивать отсутствующие высокочастотные диапазоны.
    """
    def __init__(self, loss_weight=0.5, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha # Коэффициент фокусировки на сложных (высоких) частотах
    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        amp_distance = (pred_amp - target_amp) ** 2
        max_dist = torch.max(amp_distance).detach() + 1e-8
        focal_weight = (amp_distance / max_dist) ** self.alpha
        frequency_loss = focal_weight * amp_distance
        return frequency_loss.mean() * self.loss_weight
class FocalFrequencyLossLog(nn.Module):
    """
    Focal Frequency Loss с логарифмическим сжатием амплитуд по ГОСТ и ТЗ.
    Уравновешивает вклады низких и высоких частот для стабилизации цвета.
    """
    def __init__(self, loss_weight=1.0, alpha=1.0, log_factor=100.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.gamma = log_factor  # 🎯 Задаем числовой фактор сжатия спектра
    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        pred_amp_log = torch.log(1.0 + self.gamma * pred_amp) / torch.log(torch.tensor(1.0 + self.gamma, device=pred.device))
        target_amp_log = torch.log(1.0 + self.gamma * target_amp) / torch.log(torch.tensor(1.0 + self.gamma, device=pred.device))
        amp_distance = (pred_amp_log - target_amp_log) ** 2
        max_dist = torch.max(amp_distance).detach() + 1e-8
        focal_weight = (amp_distance / max_dist) ** self.alpha
        frequency_loss = focal_weight * amp_distance

## === libraries/training_losses.py (part 2) ===
        return frequency_loss.mean() * self.loss_weight
class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        losses_cfg = config.get('losses', {})
        self.l1_weight = losses_cfg.get('l1_weight', 1.0)
        self.ffl_weight = losses_cfg.get('ffl_weight', 1.0)
        self.ffl_type = losses_cfg.get('ffl_type', 'linear')
        self.l1_loss = nn.L1Loss()
        log_factor = losses_cfg.get('ffl_log_factor', 100.0)
        if self.ffl_type == 'log':
            self.ffl_loss = FocalFrequencyLossLog(
                loss_weight=1.0,
                alpha=losses_cfg.get('ffl_alpha', 1.0),
                log_factor=log_factor
            )
        else:
            self.ffl_loss = FocalFrequencyLoss(
                loss_weight=1.0,
                alpha=losses_cfg.get('ffl_alpha', 1.0)
            )
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ffl = self.ffl_loss(pred, target)
        total = self.l1_weight * l1 + self.ffl_weight * ffl
        return total, l1.item(), ffl.item()

## === evaluate_baseline.py (part 1) ===
import sys
from libraries.config import get_pipeline_config
from libraries.evaluate_baseline import run_baseline_evaluation
from libraries.logger import setup_logger, get_logger
def main():
    config = get_pipeline_config()
    log_cfg = config.get('pipeline_logger', {})
    log_file = log_cfg.get('log_file', 'pipeline.log')
    setup_logger(log_file)   # гарантируем, что логгер настроен
    logger = get_logger()
    logger.info("=== Запуск оценки baseline (Bilinear, MHC) ===")
    run_baseline_evaluation(config)
    logger.info("=== Оценка завершена ===")
if __name__ == '__main__':
    main()

## === plot_metrics.py (part 1) ===
import os
import sys
import re
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def smooth_exponential(data, alpha):
    if alpha >= 1.0:
        return data
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed
def load_csv_data(filepath, x_col, y_col, smoothing=1.0):
    df = pd.read_csv(filepath)
    x = df[x_col].values
    y = df[y_col].values
    if smoothing < 1.0:
        y = smooth_exponential(y, smoothing)
    mask = ~np.isnan(y)
    return x[mask], y[mask]
def load_log_data(filepath, pattern, smoothing=1.0):
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                values.append(float(match.group(1)))
    if not values:
        raise ValueError(f"Не найдено значений по паттерну {pattern} в {filepath}")
    x = np.arange(1, len(values)+1)
    y = np.array(values)
    if smoothing < 1.0:
        y = smooth_exponential(y, smoothing)
    return x, y
def plot_line(ax, x, y, title, xlabel, ylabel, y_lim=None, y_scale='linear', color=None):
    ax.plot(x, y, color=color or 'blue', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_lim:
        ax.set_ylim(y_lim)
    if y_scale == 'log':
        ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5)
def plot_comparison(ax, x, y, baseline_value, baseline_label, title, xlabel, ylabel, y_lim=None):
    ax.plot(x, y, color='blue', linewidth=1.5, label='NAFNet')

## === plot_metrics.py (part 2) ===
    ax.axhline(y=baseline_value, color='red', linestyle='--', linewidth=1.5, label=baseline_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
def plot_multi_line(ax, x, y_dict, title, xlabel, ylabel, y_scale='linear', legend=None):
    for i, (key, yvals) in enumerate(y_dict.items()):
        ax.plot(x, yvals, linewidth=1.5, label=legend[i] if legend else key)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_scale == 'log':
        ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5)
    if legend or len(y_dict) > 1:
        ax.legend()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', required=True, help='Путь к YAML конфигурации графиков')
    args = parser.parse_args()
    with open(args.opt, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    exp_dir = cfg['experiment_dir']
    out_dir = cfg.get('output_dir', os.path.join(exp_dir, 'figures'))
    os.makedirs(out_dir, exist_ok=True)
    data_files = cfg.get('data_files', {})
    for plot_cfg in cfg['plots']:
        plot_type = plot_cfg.get('type', 'line')
        data_cfg = plot_cfg['data']
        source = data_cfg.get('source', 'csv')
        if source == 'val_csv':
            filepath_abs = os.path.join(exp_dir, 'val_metrics.csv')
            if not os.path.exists(filepath_abs):
                print(f"Пропуск {plot_cfg.get('save')}: файл {filepath_abs} не найден")
                continue
        elif 'file_ref' in data_cfg:
            ref = data_cfg['file_ref']
            if ref not in data_files:
                print(f"Пропуск {plot_cfg.get('save')}: неизвестный file_ref '{ref}'")
                continue
            filepath_abs = os.path.abspath(data_files[ref])
            if not os.path.exists(filepath_abs):
                print(f"Пропуск {plot_cfg.get('save')}: файл {filepath_abs} не найден")
                continue
        elif 'file' in data_cfg:
            filepath_abs = os.path.abspath(data_cfg['file'])
            if not os.path.exists(filepath_abs):

## === plot_metrics.py (part 3) ===
                print(f"Пропуск {plot_cfg.get('save')}: файл {filepath_abs} не найден")
                continue
        else:
            print(f"Пропуск {plot_cfg.get('save')}: не указан file или file_ref")
            continue
        try:
            if source == 'csv' or source == 'val_csv':
                if plot_type == 'multi_line':
                    df = pd.read_csv(filepath_abs)
                    x_col = data_cfg['x']
                    x = df[x_col].values
                    y_columns = data_cfg['y']
                    y_dict = {}
                    smoothing = plot_cfg.get('smoothing', 1.0)
                    for col in y_columns:
                        y_vals = df[col].values
                        if smoothing < 1.0:
                            y_vals = smooth_exponential(y_vals, smoothing)
                        y_dict[col] = y_vals
                    xlabel = data_cfg.get('x_label', x_col.capitalize())
                    ylabel = plot_cfg.get('y_label', 'Value')
                    fig, ax = plt.subplots(figsize=(8, 5))
                    plot_multi_line(ax, x, y_dict, plot_cfg['title'], xlabel, ylabel,
                                    y_scale=plot_cfg.get('y_scale', 'linear'),
                                    legend=plot_cfg.get('legend'))
                else:
                    x_col = data_cfg['x']
                    y_col = data_cfg['y']
                    smoothing = plot_cfg.get('smoothing', 1.0)
                    x, y = load_csv_data(filepath_abs, x_col, y_col, smoothing)
                    xlabel = data_cfg.get('x_label', x_col.capitalize())
                    ylabel = plot_cfg.get('y_label', y_col.capitalize())
                    fig, ax = plt.subplots(figsize=(8, 5))
                    if plot_type == 'line':
                        plot_line(ax, x, y, plot_cfg['title'], xlabel, ylabel,
                                  y_lim=plot_cfg.get('y_lim'),
                                  y_scale=plot_cfg.get('y_scale', 'linear'))
                    elif plot_type == 'comparison':
                        baseline_value = plot_cfg['baseline_value']
                        baseline_label = plot_cfg.get('baseline_label', 'Baseline')
                        plot_comparison(ax, x, y, baseline_value, baseline_label,
                                        plot_cfg['title'], xlabel, ylabel,
                                        y_lim=plot_cfg.get('y_lim'))
                    else:
                        raise ValueError(f"Unknown plot type for csv: {plot_type}")
            elif source == 'log':
                pattern = data_cfg['pattern']
                smoothing = plot_cfg.get('smoothing', 1.0)
                x, y = load_log_data(filepath_abs, pattern, smoothing)
                xlabel = data_cfg.get('x_label', 'Epoch')

## === plot_metrics.py (part 4) ===
                ylabel = plot_cfg.get('y_label', 'Value')
                fig, ax = plt.subplots(figsize=(8, 5))
                plot_line(ax, x, y, plot_cfg['title'], xlabel, ylabel,
                          y_lim=plot_cfg.get('y_lim'),
                          y_scale=plot_cfg.get('y_scale', 'linear'))
            else:
                raise ValueError(f"Unknown source: {source}")
            save_path = os.path.join(out_dir, plot_cfg['save'])
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Сохранён график: {save_path}")
        except Exception as e:
            print(f"Ошибка при построении {plot_cfg.get('save', 'неизвестного графика')}: {e}")
            continue
if __name__ == '__main__':
    main()

