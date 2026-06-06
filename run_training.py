#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    model = create_nafnet_model(opt, device)
    optimizer, scheduler = create_optimizer_and_scheduler(model, opt)

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
                    metrics['psnr'].item(), current_lr, global_step,
                    model=model, targets=hq, outputs=out, device=device,
                    l1_loss_val=metrics['l1_val'], ffl_loss_val=metrics['ffl_val']
                )
                global_step += 1
                # внутри цикла, после вычисления metrics
                log_progress(
                    logger=logger,
                    exp_name=opt['name'],
                    epoch=epoch,
                    total_epochs=train_cfg['num_epochs'],
                    batch_idx=batch_idx,
                    total_batches=len(train_loader),
                    loss_val=metrics['total_loss'].item(),
                    psnr_val=metrics['psnr'].item(),
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
                # При плановом сохранении
                save_checkpoint(checkpoint_path, epoch, 0, model, optimizer, scheduler, global_step, is_emergency=False)
                logger.info(f"Чекпоинт сохранён для эпохи {epoch+1}")

    except KeyboardInterrupt:
        logger.warning("Прерывание по Ctrl+C, аварийное сохранение...")
        # При аварийном сохранении (KeyboardInterrupt)
        save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=True)
        train_logger.close()
        sys.exit(0)

if __name__ == '__main__':
    main()