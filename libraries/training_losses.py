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
        # pred, target имеют размерность [B, C, H, W]
        # Вычисляем двумерное БПФ по пространственным осям (H, W)
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        # Переносим нулевую частоту в центр (для правильного анализа спектра)
        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))
        
        # Извлекаем амплитудный спектр (модуль комплексного числа)
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # Считаем базовую разность амплитуд
        amp_distance = (pred_amp - target_amp) ** 2
        
        # Матрица спектрального взвешивания (focal weights):
        # Чем сильнее сеть ошибается в частоте, тем выше вес этой частоты в лоссе
        max_dist = torch.max(amp_distance).detach() + 1e-8
        focal_weight = (amp_distance / max_dist) ** self.alpha
        
        # Итоговый частотный лосс
        frequency_loss = focal_weight * amp_distance
        
        return frequency_loss.mean() * self.loss_weight

class FocalFrequencyLossLog(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, log_factor=100.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.gamma = log_factor  # Фактор сжатия спектра
        
        # 🎯 Вычисляем константу ОДИН РАЗ при инициализации и регистрируем как буфер
        # Больше никаких блокирующих аллокаций памяти внутри forward!
        denominator = torch.log(torch.tensor(1.0 + self.gamma))
        self.register_buffer('log_denominator', denominator)

    def forward(self, pred, target):
        # Перевод в частотную область через двумерное быстрое преобразование Фурье
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        # Сдвиг низкочастотных компонент в центр спектра
        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))
        
        # Извлекаем амплитудный спектр
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # ⚡ Логарифмическое сжатие динамического диапазона частот с использованием готового буфера
        pred_amp_log = torch.log(1.0 + self.gamma * pred_amp) / self.log_denominator
        target_amp_log = torch.log(1.0 + self.gamma * target_amp) / self.log_denominator
        
        # Вычисление взвешенного частотного расстояния (Focal Loss в спектральной области)
        amp_distance = (pred_amp_log - target_amp_log) ** 2
        max_dist = torch.max(amp_distance).detach() + 1e-8
        
        focal_weight = (amp_distance / max_dist) ** self.alpha
        frequency_loss = focal_weight * amp_distance
        
        return frequency_loss.mean() * self.loss_weight

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        losses_cfg = config.get('losses', {})
        self.l1_weight = losses_cfg.get('l1_weight', 1.0)
        self.ffl_weight = losses_cfg.get('ffl_weight', 1.0)
        
        # 🎯 Извлекаем эпоху старта спектрального лосса (дефолт — 0, т.е. сразу)
        self.ffl_start_epoch = losses_cfg.get('ffl_start_epoch', 0)
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

    def forward(self, pred, target, current_epoch=0):
        """
        current_epoch: Текущая эпоха обучения, передаваемая из основного цикла.
        """
        l1 = self.l1_loss(pred, target)
        
        # 🎯 Автоматический динамический прогрев: проверяем веху эпохи
        if current_epoch >= self.ffl_start_epoch:
            ffl = self.ffl_loss(pred, target)
            active_ffl_weight = self.ffl_weight
        else:
            # Период прогрева: спектральный лосс исключён из графа вычислений
            ffl = torch.tensor(0.0, device=pred.device)
            active_ffl_weight = 0.0
            
        total = self.l1_weight * l1 + active_ffl_weight * ffl
        return total, l1.item(), ffl.item()

