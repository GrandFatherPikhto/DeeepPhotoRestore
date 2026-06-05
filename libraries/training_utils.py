import os
import torch
from torch.utils.data import DataLoader
from libraries.config import get_pipeline_config
from libraries.device import get_torch_device
from libraries.pipeline_data import create_restoration_dataset
from libraries.model_utils import create_nafnet_model
from libraries.logger import setup_logger, get_logger
# from libraries.training_checkpoint import get_checkpoint_path, load_checkpoint
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
    dataset = create_restoration_dataset(opt, is_train=True)
    train_cfg = opt['datasets']['train']
    return DataLoader(
        dataset,
        batch_size=train_cfg['batch_size_per_gpu'],
        shuffle=True,
        num_workers=train_cfg.get('num_worker_per_gpu', 4),
        pin_memory=(torch.cuda.is_available())
    )

def create_optimizer_and_scheduler(model, opt):
    train_cfg = opt['train']
    optim_cfg = train_cfg['optim_g']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg['lr']),
        weight_decay=optim_cfg.get('weight_decay', 0.0),
        betas=optim_cfg.get('betas', (0.9, 0.999))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['num_epochs'],
        eta_min=float(train_cfg.get('scheduler', {}).get('eta_min', 1e-7))
    )
    return optimizer, scheduler

def cleanup_experiment(opt, save_dir, logger):
    if opt.get('clean_training', False):
        import shutil
        if os.path.exists(save_dir):
            logger.info(f"Очистка эксперимента: {save_dir}")
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        return True
    return False

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
    return 20 * torch.log10(1.0 / torch.sqrt(mse_loss))

def compute_metrics(out, target, criterion, loss_type):
    """
    Вычисляет loss, компоненты L1/FFL и PSNR.
    Возвращает словарь с ключами: total_loss, l1_val, ffl_val, psnr.
    """
    if loss_type == 'combined':
        total_loss, l1_val, ffl_val = criterion(out, target)
    else:
        total_loss = criterion(out, target)
        l1_val = total_loss.item()
        ffl_val = 0.0

    mse = nn.MSELoss()(out, target).detach()
    psnr = torch.tensor(calculate_psnr(mse))   # превращаем float в тензор
    return {
        'total_loss': total_loss,
        'l1_val': l1_val,
        'ffl_val': ffl_val,
        'psnr': psnr
    }