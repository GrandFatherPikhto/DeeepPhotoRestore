#!./.venv/bin/python
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

# Импортируем наши чистые структурированные библиотеки
from libraries.environment import init_environment
from libraries.checkpoint_io import get_checkpoint_path, load_checkpoint, save_checkpoint
from libraries.model_loader import load_network
from libraries.validator import VisualValidator
from libraries.logger import TrainingLogger

# --- ФУНКЦИЯ РАСЧЕТА PSNR ---
def calculate_psnr(mse_loss):
    if mse_loss == 0: return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse_loss))

# --- ЛОКАЛЬНЫЙ ПАРНЫЙ ДАТАСЕТ ---
class JPEGKeyPairDataset(Dataset):
    def __init__(self, lq_dir, hq_dir, transform):
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(lq_dir) if f.endswith('.jpg')]
        self.lq_dir = lq_dir
        self.hq_dir = hq_dir
        self.transform = transform
    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        name = self.filenames[idx]
        lq_img = Image.open(os.path.join(self.lq_dir, f"{name}.jpg")).convert('RGB')
        hq_img = Image.open(os.path.join(self.hq_dir, f"{name}.png")).convert('RGB')
        return self.transform(lq_img), self.transform(hq_img)


def main():
    # 1. Наводим порядок в окружении
    args, opt, save_dir, device = init_environment()
    exp_name = opt.get('name', 'default_exp')
    
    # 2. Инициализация Данных
    dataset_root = opt['path']['dataset_root']
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    # 2. Инициализация Данных (JPEG или RAW)
    dataset_opt = opt['datasets']['train']
    dataset_type = dataset_opt['type']
    dataset_root = opt['path']['dataset_root']
    
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    
    # 🎯 УНИВЕРСАЛЬНЫЙ ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМОВ ДАТАСЕТА:
    if dataset_type in ['PairedImageDataset', 'JPEGKeyPairDataset']:
        print("📦 Загружаем локальный датасет для JPEG...")
        train_dataset = JPEGKeyPairDataset(
            lq_dir=os.path.join(dataset_root, "train/lq_inputs"),
            hq_dir=os.path.join(dataset_root, "train/hq_targets"),
            transform=transform
        )
    elif dataset_type == 'CustomNEFPairDataset':
        print("📦 Загружаем специализированный датасет для NEF (RAW)...")
        # Подгружаем класс из файла libraries/dataset_nef.py
        from libraries.dataset_nef import CustomNEFPairDataset
        train_dataset = CustomNEFPairDataset(
            dataroot_lq=os.path.join(dataset_root, "train/lq_inputs"),
            dataroot_gt=os.path.join(dataset_root, "train/hq_targets")
        )
    else:
        raise ValueError(f"Неизвестный тип датасета в YAML: {dataset_type}")

    num_workers = dataset_opt.get('num_worker_per_gpu', 0)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=dataset_opt['batch_size_per_gpu'], 
        shuffle=True,
        num_workers=num_workers, # 🎯 Передаем потоки сюда
        pin_memory=True if torch.cuda.is_available() else False # Ускорит передачу тензоров в VRAM
    )        
    # train_loader = DataLoader(train_dataset, batch_size=dataset_opt['batch_size_per_gpu'], shuffle=True)

    # 3. Инициализация Компонентов Сети
    model = load_network(opt, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt['train']['optim_g']['lr']))
    criterion = nn.L1Loss() # Наш злой L1 лосс против мыла!

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt['train']['num_epochs'], eta_min=float(opt.get('train', {}).get('scheduler', {}).get('eta_min', 1e-7))
    )

    # 4. Инициализация Логгера и Валидатора
    logger = TrainingLogger(opt, save_dir)
    visual_validator = VisualValidator(opt)
    checkpoint_path = get_checkpoint_path(args, opt)

    # 5. Загрузка Чекпоинта (передаем аргумент clean, чтобы сбросить загрузку при очистке)
    start_epoch, start_batch, global_step = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, train_loader, device, 
        auto_resume=opt['datasets']['train']['auto_resume'], ignore_resume=args.clean
    )

    # 6. Главный Тренировочный Цикл
    try:
        save_every_n_epochs = opt.get('train', {}).get('save_checkpoint_epoch', 10)
        batch_idx = start_batch # Защита от моментального Ctrl+C

        for epoch in range(start_epoch, opt['train']['num_epochs']):
            model.train()
            for batch_idx, (lq_images, hq_images) in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue
                
                inputs, targets = lq_images.to(device), hq_images.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, dict): outputs = outputs['out']
                    
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Логирование
                psnr_value = calculate_psnr(nn.MSELoss()(outputs, targets).detach())
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.log_metrics(loss.item(), psnr_value.item(), current_lr, global_step, model, targets, outputs, device)
                global_step += 1            
                
                # === НАСТРАИВАЕМЫЙ ВЫВОД В КОНСОЛЬ ===
                logger_opt = opt.get('logger', {})
                if batch_idx % logger_opt.get('print_freq', 2) == 0:
                    console_flags = logger_opt.get('include_in_console', {})
                    console_parts = [f"[{exp_name}] Эп [{epoch}/{opt['train']['num_epochs']}] Бт [{batch_idx}/{len(train_loader)}]"]
                    if console_flags.get('loss', True): console_parts.append(f"L1 Лосс: {loss.item():.5f}")
                    if console_flags.get('psnr', True): console_parts.append(f"PSNR: {psnr_value.item():.2f} dB")
                    if console_flags.get('vram', True) and torch.cuda.is_available():
                        console_parts.append(f"VRAM: {torch.cuda.memory_allocated(device)/(1024**3):.2f}GB")
                    print(" | ".join(console_parts))
            
            # Конец эпохи
            start_batch = 0  
            scheduler.step()

            # Плановое сохранение и Валидация
            if (epoch + 1) % save_every_n_epochs == 0:
                save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=False)
                print(f"\n💾 [Автосохранение] Чекпоинт обновлен для старта Эпохи {epoch + 1}")
                visual_validator.run_validation(model, epoch, device)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Прерывание процесса! Аварийное сохранение состояния...")
        save_checkpoint(checkpoint_path, epoch, batch_idx, model, optimizer, scheduler, global_step, is_emergency=True)
        logger.close()
        print(f"Чекпоинт успешно обновлен по пути: {checkpoint_path}")
        sys.exit(0)

if __name__ == "__main__":
    main()
