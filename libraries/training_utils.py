import os
import torch
import shutil
import math
from torch.utils.data import DataLoader
from libraries.config import get_pipeline_config
from libraries.device import get_torch_device
from libraries.pipeline_data import create_restoration_dataset
from libraries.model_utils import create_nafnet_model
from libraries.logger import setup_logger, get_logger
# from libraries.training_checkpoint import get_checkpoint_path, load_checkpoint
import torch.nn as nn

# Инициализируем оператор один раз на уровне модуля, чтобы не плодить сущности на каждом батче
_mse_criterion = nn.MSELoss()

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
    # Извлекаем пути и параметры датасета из общего конфига
    path_cfg = opt['path']
    train_cfg = opt['datasets']['train']
    
    dataroot_lq = os.path.join(path_cfg['dataset_root'], 'train', 'lq_inputs')
    dataroot_gt = os.path.join(path_cfg['dataset_root'], 'train', 'hq_targets')
    
    # Передаем конфигурацию opt целиком в конструктор датасета
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
        model: torch.nn.Module
        opt (dict): полный конфигурационный словарь (с секциями 'train' и 'scheduler')
        total_training_steps (int, optional): общее количество итераций (шагов) для OneCycleLR.
            Если не указан, будет вычислен как num_epochs * len(train_loader).
    
    Returns:
        optimizer, scheduler
    """
    train_cfg = opt['train']
    optim_cfg = train_cfg['optim_g']
    
    # Создаём оптимизатор AdamW (можно легко заменить на другой тип, если нужно)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg['lr']),
        weight_decay=float(optim_cfg.get('weight_decay', 0.0)),
        betas=optim_cfg.get('betas', (0.9, 0.999))
    )
    
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', 'CosineAnnealingLR')
    
    # ========== ДОСТУПНЫЕ ТИПЫ ПЛАНИРОВЩИКОВ ==========
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
        # ВНИМАНИЕ: этот планировщик требует вызова scheduler.step(val_loss) в цикле валидации
        mode = scheduler_cfg.get('mode', 'min')
        factor = float(scheduler_cfg.get('factor', 0.1))
        patience = int(scheduler_cfg.get('patience', 10))
        threshold = float(scheduler_cfg.get('threshold', 1e-4))
        cooldown = int(scheduler_cfg.get('cooldown', 0))
        min_lr = float(scheduler_cfg.get('min_lr', 0.0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            threshold=threshold, cooldown=cooldown, min_lr=min_lr
        )
    
    elif scheduler_type == 'OneCycleLR':
        # Требует общего количества шагов (total_steps)
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
        # lr_lambda – можно передать как строку (eval) или готовую функцию
        lr_lambda = scheduler_cfg.get('lr_lambda')
        if lr_lambda is None:
            raise KeyError("LambdaLR requires 'lr_lambda' in scheduler config")
        if isinstance(lr_lambda, str):
            # Осторожно: eval может быть опасен, лучше передавать имя функции
            # Для простоты предполагаем, что пользователь предоставил лямбда-выражение
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

def calculate_psnr(mse):
    """
    Безопасный расчёт PSNR из тензора MSE.
    """
    # Переводим тензор в число CPU, чтобы избежать ошибок ветвления PyTorch
    mse_val = mse.item() if isinstance(mse, torch.Tensor) else mse
    
    if mse_val == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse_val))


def compute_metrics(out, target, criterion, loss_type, current_epoch=0):
    """
    Вычисляет комбинированный лосс и оперативные метрики для текущего батча
    с учётом временной фазы прогрева графа вычислений.
    """
    # 1. Обсчёт целевых функций потерь с учётом эпохи прогрева
    if loss_type == 'combined':
        total_loss, l1_val, ffl_val = criterion(out, target, current_epoch=current_epoch)
    else:
        total_loss = criterion(out, target)
        l1_val = total_loss.item()
        ffl_val = 0.0

    # 2. Прецизионный расчёт оперативного PSNR через глобальный экземпляр
    # Извлекаем чистое число через .item(), полностью ликвидируя риски булевой двусмысленности тензоров!
    mse_tensor = _mse_criterion(out, target).detach()
    psnr_val = calculate_psnr(mse_tensor.item())  # Передаём чистый float

    # 3. Собираем монолитный админский словарь метрик для run_training.py
    metrics = {
        'total_loss': total_loss,
        'l1_loss': l1_val,
        'ffl_loss': ffl_val,
        'psnr': psnr_val
    }
    
    return metrics



def cleanup_experiment(opt, save_dir, logger):
    if not opt.get('clean_training', False):
        return False

    logger.info(f"Выборочная очистка файлов обучения в {save_dir} (оставляем debug_visuals и pipeline.log)")

    # Удаляем папки, относящиеся к обучению
    for subdir in ['checkpoints', 'tb_logs', 'val_predictions']:
        subdir_path = os.path.join(save_dir, subdir)
        if os.path.exists(subdir_path):
            shutil.rmtree(subdir_path)
            logger.info(f"Удалена папка: {subdir_path}")

    # Удаляем файлы метрик обучения
    for filename in ['train_metrics.csv', 'train_progress.log', 'val_metrics.csv']:
        file_path = os.path.join(save_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Удалён файл: {file_path}")

    # Папку debug_visuals и файл pipeline.log НЕ удаляем
    # (они могут быть созданы run_pipeline.py)

    # Пересоздаём основные папки, чтобы избежать ошибок при записи
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'tb_logs'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val_predictions'), exist_ok=True)

    return True