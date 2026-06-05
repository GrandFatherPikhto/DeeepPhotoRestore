import torch
import torch.nn as nn

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
    """
    Focal Frequency Loss с логарифмическим сжатием амплитуд.
    Использует log(1 + amplitude) для выравнивания вклада низких и высоких частот.
    """
    def __init__(self, loss_weight=0.5, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        # pred, target: [B, C, H, W]
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        # Логарифмическое сжатие (избегаем log(0) добавлением 1)
        pred_amp_log = torch.log(1 + pred_amp)
        target_amp_log = torch.log(1 + target_amp)

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
        self.ffl_type = losses_cfg.get('ffl_type', 'linear')  # 'linear' или 'log'
        self.l1_loss = nn.L1Loss()

        # Выбираем тип FFL
        if self.ffl_type == 'log':
            self.ffl_loss = FocalFrequencyLossLog(
                loss_weight=1.0,
                alpha=losses_cfg.get('ffl_alpha', 1.0)
            )
        else:
            self.ffl_loss = FocalFrequencyLoss(
                loss_weight=1.0,
                alpha=losses_cfg.get('ffl_alpha', 1.0)
            )

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ffl = self.ffl_loss(pred, target)
        total = self.l1_weight * l1 + self.ffl_weight * ffl
        return total, l1.item(), ffl.item()