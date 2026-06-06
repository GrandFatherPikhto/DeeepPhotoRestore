#!/./.venv/bin/python
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from libraries.logger import get_logger

logger = get_logger()

try:
    from libraries.training_losses import FocalFrequencyLoss
    HAS_FFL = True
except ModuleNotFoundError:
    logger.warning("FocalFrequencyLoss не найден! Используется только L1-Loss.")
    HAS_FFL = False

def run_smoke_test(model, dataset, config, device):
    """
    Нагрузочное тестирование градиентных потоков и видеопамяти GPU.
    Проверяет связку: NAFNet + FocalFrequencyLoss + AdamW.
    """
    logger.info("=== Запуск модуля стресс-тестирования (Smoke Test) ===")
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Вычислительное устройство: {device}")
    
    model = model.to(device)
    model.train()
    
    l1_loss_fn = nn.L1Loss()
    if HAS_FFL:
        ffl_cfg = config.get("losses", {}).get("ffl_opt", {})
        ffl_alpha = ffl_cfg.get("alpha", 1.0)
        ffl_weight = ffl_cfg.get("loss_weight", 1.0)
        ffl_loss_fn = FocalFrequencyLoss(loss_weight=ffl_weight, alpha=ffl_alpha)
        logger.info(f"Инициализирован FocalFrequencyLoss (weight={ffl_weight})")
    
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 1. Извлекаем и подготавливаем батч данных
    try:
        lq_tensor, hq_tensor = dataset[0] # Берем первый сэмпл датасета
        lq_batch = lq_tensor.unsqueeze(0).to(device)
        hq_batch = hq_tensor.unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Не удалось подготовить тестовый батч данных: {e}")
        raise e
        
    # 2. Вычисления на графе нейросети
    try:
        optimizer.zero_grad()
        
        # Прямой проход (Выход pred_batch создается ТУТ)
        pred_batch = model(lq_batch)
        logger.info(f"Прямой проход успешен. Размерность выхода: {list(pred_batch.shape)}")
        
        # Защитная интерполяция маски СТРОГО ПОСЛЕ создания pred_batch
        if hq_batch.shape[2:] != pred_batch.shape[2:]:
            logger.warning(f"⚠️ Масштаб таргета {list(hq_batch.shape)} не совпадает с предсказанием {list(pred_batch.shape)}. Интерполируем.")
            hq_batch = F.interpolate(hq_batch, size=(pred_batch.shape[2], pred_batch.shape[3]), mode='bilinear', align_corners=False)
            
        # Расчет комбинированного лосса
        loss = l1_loss_fn(pred_batch, hq_batch)
        if HAS_FFL:
            loss += ffl_loss_fn(pred_batch, hq_batch)
            
        logger.info(f"Расчет лосса успешен. Значение: {loss.item():.4f}")
        
        # Обратный проход
        loss.backward()
        optimizer.step()
        logger.info("Обратный проход градиентов и шаг AdamW выполнены без ошибок.")
        logger.info("🚀 Стендовый нагрузочный тест (Smoke Test) успешно пройден!")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Out of Memory на GPU!")
        else:
            logger.error(f"Сбой во время вычислений на графе нейросети: {e}")
        raise e
