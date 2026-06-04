import os
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, opt, save_dir):
        self.opt = opt
        self.save_dir = save_dir
        self.use_tb = opt.get('logger', {}).get('use_tb_logger', False)
        self.tb_logger = None
        
        # Инициализируем TensorBoard только если это включено в YAML
        if self.use_tb:
            log_dir = os.path.join(save_dir, 'tb_logs')
            self.tb_logger = SummaryWriter(log_dir=log_dir)
            print(f"📊 [Логгер] TensorBoard инициализирован в: {log_dir}")
        else:
            print("ℹ️  [Логгер] TensorBoard отключен в настройках YAML.")

    def log_metrics(self, loss_val, psnr_val, lr_val, global_step, model=None, targets=None, outputs=None, device=None):
        """
        Расширенный метод логирования с динамической сборкой строки.
        Принимает модель и тензоры для расчета расширенной телеметрии.
        """
        # 1. Запись базовых параметров в TensorBoard (🎯 ИСПРАВЛЕНО: имя tb_logger вместо tb_writer)
        if self.use_tb and self.tb_logger is not None:
            self.tb_logger.add_scalar('train/loss', loss_val, global_step)
            self.tb_logger.add_scalar('train/psnr', psnr_val, global_step)
            self.tb_logger.add_scalar('train/lr', lr_val, global_step)

        # 2. Чтение настроек из YAML для текстового лога
        logger_opt = self.opt.get('logger', {})
        file_freq = logger_opt.get('file_log_freq', 1)
        
        # Запись в файл по расписанию
        if global_step % file_freq == 0:
            import os
            import torch
            import numpy as np
            
            log_file_name = logger_opt.get('log_file_name', 'train_progress.log')
            log_path = os.path.join(self.save_dir, log_file_name)
            flags = logger_opt.get('include_in_file_log', {})
            
            log_parts = [f"[Шаг: {global_step:05d}]"]
            
            # Разбираем флаги из YAML-конфига
            if flags.get('loss', True):
                log_parts.append(f"Loss: {loss_val:.6f}")
            if flags.get('psnr', True):
                log_parts.append(f"PSNR: {psnr_val:.2f} dB")
            if flags.get('learning_rate', True):
                log_parts.append(f"LR: {lr_val:.2e}")
                
            # Расчет Delta PSNR (индикатор выхода на ленивое плато)
            if flags.get('delta_psnr', True):
                current_psnr = psnr_val
                if not hasattr(self, 'prev_psnr'):
                    self.prev_psnr = current_psnr
                delta = current_psnr - self.prev_psnr
                self.prev_psnr = current_psnr
                log_parts.append(f"dPSNR: {delta:+.4f}")
                
            # Расчет Дисперсии Градиентов (индикатор путаницы сети в JPEG/RAW)
            if flags.get('grad_variance', True) and model is not None:
                # 🎯 ДОБАВЛЕНО: Безопасная проверка p.grad is not None, чтобы избежать падения на первом шаге
                grad_norms = [p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None]
                grad_var = np.var(grad_norms) if grad_norms else 0.0
                log_parts.append(f"GradVar: {grad_var:.2e}")
                
            # Расчет TV Ratio (детектор появления кубиков или мыла)
            if flags.get('tv_ratio', True) and targets is not None and outputs is not None:
                tv_out = torch.sum(torch.abs(outputs[:, :, :, :-1] - outputs[:, :, :, 1:])) + \
                         torch.sum(torch.abs(outputs[:, :, :-1, :] - outputs[:, :, 1:, :]))
                tv_tar = torch.sum(torch.abs(targets[:, :, :, :-1] - targets[:, :, :, 1:])) + \
                         torch.sum(torch.abs(targets[:, :, :-1, :] - targets[:, :, 1:, :]))
                tv_ratio = (tv_out / (tv_tar + 1e-8)).item()
                log_parts.append(f"TV_Ratio: {tv_ratio:.3f}")
                
            # Телеметрия видеокарты RTX 4070Ti
            if flags.get('vram_usage', True) and device is not None and torch.cuda.is_available():
                alloc_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
                res_vram = torch.cuda.memory_reserved(device) / (1024 ** 3)
                log_parts.append(f"VRAM: {alloc_vram:.2f}GB/{res_vram:.2f}GB")
                
            # Склеиваем строку и дописываем в файл прогресса
            log_string = " | ".join(log_parts) + "\n"
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_string)


    def log_validation_image(self, tag, image_tensor, epoch):
        """
        Задел на будущее: если захотим отправлять матрицы картинок 
        прямо в браузер на вкладку Images
        """
        if self.use_tb and self.tb_logger:
            # image_tensor должен быть в формате [C, H, W] и в диапазоне [0, 1]
            self.tb_logger.add_image(tag, image_tensor, epoch)

    def close(self):
        """Безопасное закрытие потоков записи при выходе или Ctrl+C"""
        if self.tb_logger:
            self.tb_logger.close()
            print("📊 [Логгер] Потоки TensorBoard успешно закрыты.")
