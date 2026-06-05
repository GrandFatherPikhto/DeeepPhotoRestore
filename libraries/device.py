# -*- coding: utf-8 -*-
"""
Модуль определения вычислительного устройства (CPU/CUDA).
"""

import torch

def get_torch_device():
    """
    Определяет доступное устройство: CUDA (GPU) или CPU.
    Возвращает объект torch.device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Опционально: вывод информации об устройстве
        print(f"🔧 [Device] Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("🔧 [Device] CUDA не найдена, используется CPU")
    return device

def move_model_to_device(model, device):
    """Переносит модель на указанное устройство."""
    return model.to(device)