#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
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

def calculate_psnr(mse_loss):
    if mse_loss == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse_loss))

def main():
    config = get_pipeline_config()
    opt = config

    exp_name = opt.get('name', 'default_exp')
    save_dir = os.path.join('experiments', exp_name)
    device = get_torch_device()

    ignore_resume = False
    if opt.get('clean_training', False):
        import shutil
        if os.path.exists(save_dir):
            print(f"🧹 Очистка эксперимента: {save_dir}")
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        ignore_resume = True

    train_dataset = create_restoration_dataset(opt, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt['datasets']['train']['batch_size_per_gpu'],
        shuffle=True,
        num_workers=opt['datasets']['train'].get('num_worker_per_gpu', 4),
        pin_memory=(device.type == 'cuda')
    )

    model = create_nafnet_model(opt, device)

    train_cfg = opt['train']
    optim_g_cfg = train_cfg['optim_g']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_g_cfg['lr']),
        weight_decay=optim_g_cfg.get('weight_decay', 0.0),
        betas=optim_g_cfg.get('betas', (0.9, 0.999))
    )

    criterion = nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['num_epochs'],
        eta_min=float(train_cfg.get('scheduler', {}).get('eta_min', 1e-7))
    )

    logger = TrainingLogger(opt, save_dir)
    visual_validator = VisualValidator(opt)
    checkpoint_path = get_checkpoint_path(opt)

    start_epoch, start_batch, global_step = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, train_loader, device,
        auto_resume=opt['datasets']['train'].get('auto_resume', True),
        ignore_resume=ignore_resume
    )

    try:
        save_every_n_epochs = train_cfg.get('save_checkpoint_epoch', 10)
        batch_idx = start_batch

        for epoch in range(start_epoch, train_cfg['num_epochs']):
            model.train()
            for batch_idx, (lq_images, hq_images) in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                inputs, targets = lq_images.to(device), hq_images.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    outputs = outputs['out']

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                psnr_value = calculate_psnr(nn.MSELoss()(outputs, targets).detach())
                current_lr = optimizer.param_groups[0]['lr']

                logger.log_metrics(
                    loss.item(), psnr_value.item(), current_lr, global_step,
                    model=model, targets=targets, outputs=outputs, device=device
                )
                global_step += 1

                logger_opt = opt.get('logger', {})
                if batch_idx % logger_opt.get('print_freq', 10) == 0:
                    console_flags = logger_opt.get('include_in_console', {})
                    parts = [f"[{exp_name}] Epoch {epoch}/{train_cfg['num_epochs']} Batch {batch_idx}/{len(train_loader)}"]
                    if console_flags.get('loss', True):
                        parts.append(f"Loss: {loss.item():.5f}")
                    if console_flags.get('psnr', True):
                        parts.append(f"PSNR: {psnr_value.item():.2f} dB")
                    if console_flags.get('vram', True) and device.type == 'cuda':
                        parts.append(f"VRAM: {torch.cuda.memory_allocated(device)/(1024**3):.2f}GB")
                    print(" | ".join(parts))

            start_batch = 0
            scheduler.step()

            if (epoch + 1) % save_every_n_epochs == 0:
                save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=False)
                print(f"\n💾 [Автосохранение] Чекпоинт для эпохи {epoch+1}")
                visual_validator.run_validation(model, epoch, device)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Аварийное сохранение...")
        save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=True)
        logger.close()
        sys.exit(0)

if __name__ == '__main__':
    main()