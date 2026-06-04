#!./.venv/bin/python
# -*- coding: utf-8 -*-

import sys
# Сначала импортируем только базовые легковесные вещи
from libraries.pipeline_config import get_pipeline_config
from libraries.pipeline_logger import setup_logger

def main():
    # Получаем конфиг, чтобы вытащить путь к логу
    config = get_pipeline_config()
    
    # Забираем путь из правильной секции pipeline_logger
    log_cfg = config.get("pipeline_logger", {})
    log_path = log_cfg.get("log_file", "pipeline.log")
    
    # Инициализируем систему логирования по нужному пути
    logger = setup_logger(log_path)
    
    logger.info("==================================================")
    logger.info(f"🚀 СТАРТ КОНВЕЙЕРА. ЛОГ ИДЕТ В: {log_path}")
    logger.info("==================================================")

    try:
        # Лениво импортируем тяжелые модули данных и обучения
        from libraries.pipeline_data import RestorationDataset
        from libraries.pipeline_visuals import run_visual_control
        from libraries.pipeline_smoke import run_smoke_test
        
        # Шаг данных с прогресс-баром
        dataset = RestorationDataset(config, is_train=True)
        
        # Шаг выборочной визуализации (картинки полетят в samples/debug_visuals)
        run_visual_control(dataset, config)
        
        # Шаг сборки модели NAFNet
        logger.info("Инициализация архитектуры нейросети NAFNet...")
        from libraries.modeling import NAFNet
        
        net_cfg = config.get("network_g", {})
        model = NAFNet(**net_cfg)
        logger.info("Экземпляр модели NAFNet успешно создан в памяти.")
        
        # Нагрузочный стресс-тест градиентов
        test_sample = dataset
        run_smoke_test(model, test_sample, config)
        
        logger.info("==================================================")
        logger.info("🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("==================================================")

    except Exception as e:
        logger.critical(f"💥 АВАРИЙНОЕ ЗАВЕРШЕНИЕ КОНВЕЙЕРА: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
