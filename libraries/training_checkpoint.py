import os
import torch
from pathlib import Path

def get_checkpoint_path(opt):
    """Определяет путь к файлу чекпоинта на основе конфига и аргумента --resume."""
    resume = opt.get('resume')  # из аргументов командной строки (--resume)
    if resume:
        return Path(resume)
    # Иначе используем resume_path из конфига (секция path) или значение по умолчанию
    resume_path = opt.get('path', {}).get('resume_path', 'checkpoints/resume.pth')
    return Path(resume_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, train_loader, device, auto_resume, ignore_resume=False):
    """Безопасно загружает состояние и вычисляет global_step"""
    start_epoch, start_batch, global_step = 0, 0, 0
    
    if auto_resume and not ignore_resume and os.path.exists(checkpoint_path):
        print(f"🔄 Обнаружен чекпоинт: {checkpoint_path}! Восстанавливаем состояние...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model')))
        optimizer.load_state_dict(ckpt.get('optimizer_state_dict', ckpt.get('opt')))
        
        start_epoch = ckpt.get('epoch', 0)
        start_batch = ckpt.get('batch_idx', ckpt.get('batch', 0)) + 1
        
        if 'scheduler_state_dict' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            
        global_step = start_epoch * len(train_loader) + start_batch
        print(f"Продолжаем с Эпохи: {start_epoch}, Батча: {start_batch} (Шаг: {global_step})")
    else:
        print("ℹ️ Чекпоинт не найден или проигнорирован. Старт с чистого листа.")
        
    return start_epoch, start_batch, global_step

def save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=False):
    """Универсальный метод сохранения (плановый или аварийный)"""
    save_path = Path(checkpoint_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_emergency:
        # При аварии фиксируем точную точку остановки
        save_dict = {
            'epoch': epoch,
            'batch': batch_idx,
            'batch_idx': batch_idx,
            'model': model.state_dict(),
            'opt': optimizer.state_dict()
        }
    else:
        # При плановом сохранении готовим старт со СЛЕДУЮЩЕЙ эпохи и нулевого батча
        save_dict = {
            'epoch': epoch + 1,
            'batch': 0,
            'batch_idx': 0,
            'model': model.state_dict(),
            'opt': optimizer.state_dict()
        }
        
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(save_dict, str(save_path))
