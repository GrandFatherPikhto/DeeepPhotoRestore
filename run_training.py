#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libraries.pipeline_config import get_pipeline_config
from libraries.pipeline_device import get_torch_device
from libraries.pipeline_data import create_restoration_dataset
from libraries.pipeline_model import create_nafnet_model
from libraries.training_logger import TrainingLogger
from libraries.training_validator import VisualValidator
from libraries.training_checkpoint import get_checkpoint_path, load_checkpoint, save_checkpoint
from libraries.pipeline_logger import get_logger, setup_logger

logger = get_logger()   # единый логгер для всего скрипта

def calculate_psnr(mse_loss):
    if mse_loss == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse_loss))

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

def cleanup_experiment(opt, save_dir):
    if opt.get('clean_training', False):
        if os.path.exists(save_dir):
            logger.info(f"Очистка эксперимента: {save_dir}")
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        return True
    return False

def init_logger(opt):
    log_cfg = opt.get('pipeline_logger', {})
    log_file = log_cfg.get('log_file', 'pipeline.log')
    setup_logger(log_file)   # импортировать из libraries.pipeline_logger
    logger = get_logger()

    return logger

def main():
    opt = get_pipeline_config()

    logger = init_logger(opt)

    exp_name = opt.get('name', 'default_exp')
    save_dir = os.path.join('experiments', exp_name)
    device = get_torch_device()

    ignore_resume = cleanup_experiment(opt, save_dir)

    train_loader = create_train_loader(opt)
    model = create_nafnet_model(opt, device)
    optimizer, scheduler = create_optimizer_and_scheduler(model, opt)

    logger.info("Инициализация логгеров и валидатора")
    train_logger = TrainingLogger(opt, save_dir)
    validator = VisualValidator(opt)
    checkpoint_path = get_checkpoint_path(opt)

    start_epoch, start_batch, global_step = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, train_loader, device,
        auto_resume=opt['datasets']['train'].get('auto_resume', True),
        ignore_resume=ignore_resume
    )

    criterion = nn.L1Loss()   # позже заменить на CombinedLoss
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

                loss = criterion(out, hq)
                loss.backward()
                optimizer.step()

                psnr = calculate_psnr(nn.MSELoss()(out, hq).detach())
                current_lr = optimizer.param_groups[0]['lr']
                train_logger.log_metrics(
                    loss.item(), psnr.item(), current_lr, global_step,
                    model=model, targets=hq, outputs=out, device=device
                )
                global_step += 1

                # Консольный вывод через единый логгер
                if batch_idx % opt.get('logger', {}).get('print_freq', 10) == 0:
                    parts = [f"[{exp_name}] Epoch {epoch}/{train_cfg['num_epochs']} Batch {batch_idx}/{len(train_loader)}"]
                    parts.append(f"Loss: {loss.item():.5f}")
                    parts.append(f"PSNR: {psnr.item():.2f} dB")
                    if device.type == 'cuda':
                        parts.append(f"VRAM: {torch.cuda.memory_allocated(device)/(1024**3):.2f}GB")
                    logger.info(" | ".join(parts))

            start_batch = 0
            scheduler.step()

            # Валидация
            val_freq = train_cfg.get('validation_freq', None)
            if val_freq is not None and (epoch + 1) % val_freq == 0:
                validator.run_validation(model, epoch, device)

            if (epoch + 1) % save_every == 0:
                save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=False)
                logger.info(f"Чекпоинт сохранён для эпохи {epoch+1}")

    except KeyboardInterrupt:
        logger.warning("Прерывание по Ctrl+C, аварийное сохранение...")
        save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=True)
        train_logger.close()
        sys.exit(0)

if __name__ == '__main__':
    main()