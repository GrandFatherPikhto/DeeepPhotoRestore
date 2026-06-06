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
    # Иначе используем resume_path из конфига (секция path) или значение по умолчанию
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
            # Аварийное сохранение: продолжаем со следующего батча
            start_batch = saved_batch + 1
        else:
            # Плановое сохранение: начинаем с начала эпохи
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
    
    # Для планового сохранения также можно сохранять epoch+1, но проще хранить текущую эпоху
    # и при загрузке не прибавлять к ней 1. Главное — единообразие.
    torch.save(save_dict, str(save_path))
