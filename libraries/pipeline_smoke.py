import torch
import torch.nn as nn
from libraries.pipeline_logger import get_logger

logger = get_logger()

# Пытаемся импортировать кастомный частотный лосс из твоих библиотек
try:
    from libraries.losses import FocalFrequencyLoss
    HAS_FFL = True
except ModuleNotFoundError:
    logger.warning("FocalFrequencyLoss не найден в libraries.losses! Будет использован заглушечный L1-Loss.")
    HAS_FFL = False

def run_smoke_test(model, dataset, config):
    """
    Выполняет нагрузочное тестирование градиентных потоков и видеопамяти GPU.
    Проверяет связку: NAFNet + FocalFrequencyLoss + AdamW.
    """
    logger.info("=== Запуск модуля стресс-тестирования (Smoke Test) ===")
    
    # 1. Настройка вычислительного устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Вычислительное устройство: {device}")
    
    # Переносим модель на GPU и переводим в режим обучения
    model = model.to(device)
    model.train()
    
    # 2. Инициализация критериев оптимизации
    l1_loss_fn = nn.L1Loss()
    if HAS_FFL:
        # Берем параметры альфы из конфига или ставим дефолтную 1.0
        ffl_cfg = config.get("losses", {}).get("ffl_opt", {})
        ffl_alpha = ffl_cfg.get("alpha", 1.0)
        ffl_weight = ffl_cfg.get("loss_weight", 1.0)
        ffl_loss_fn = FocalFrequencyLoss(loss_weight=ffl_weight, alpha=ffl_alpha)
        logger.info(f"Инициализирован FocalFrequencyLoss (weight={ffl_weight}, alpha={ffl_alpha})")
    
    # Инициализируем AdamW по параметрам твоего yml
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 3. Извлечение тестового батча
    try:
        lq_tensor, hq_tensor = dataset[0]
        # Добавляем фейковую размерность батча [B=1, C, H, W] и кидаем на GPU
        lq_batch = lq_tensor.unsqueeze(0).to(device)
        hq_batch = hq_tensor.unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Не удалось подготовить тестовый батч данных: {e}")
        raise e
        
    # 4. Прогон итерации (Forward - Backward pass)
    try:
        optimizer.zero_grad()
        
        # Прямой проход через сеть NAFNet
        pred_batch = model(lq_batch)
        logger.info(f"Прямой проход (Forward pass) успешен. Размерность выхода: {list(pred_batch.shape)}")
        
        # Расчет комбинированной функции потерь
        loss = l1_loss_fn(pred_batch, hq_batch)
        if HAS_FFL:
            loss += ffl_loss_fn(pred_batch, hq_batch)
            
        logger.info(f"Расчет комбинированного лосса успешен. Значение: {loss.item():.4f}")
        
        # Обратный проход (Градиенты)
        loss.backward()
        optimizer.step()
        logger.info("Обратный проход градиентов (Backward pass) и шаг AdamW выполнены без ошибок.")
        
        logger.info("🚀 Стендовый нагрузочный тест (Smoke Test) успешно пройден!")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Out of Memory на GPU! Уменьши размер патчей (gt_size) в yml.")
        else:
            logger.error(f"Сбой во время вычислений на графе нейросети: {e}")
        raise e
