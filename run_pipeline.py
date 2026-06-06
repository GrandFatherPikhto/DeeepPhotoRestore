#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from libraries.config import get_pipeline_config
from libraries.logger import setup_logger, get_logger
from libraries.device import get_torch_device
from libraries.pipeline_prepare import ensure_dataset_ready
from libraries.training_dataset_nef import CustomNEFPairDataset
from libraries.model_utils import create_nafnet_model
from libraries.pipeline_visuals import run_visual_control
from libraries.pipeline_smoke import run_smoke_test

def main():
    config = get_pipeline_config()
    opt_path = config.get("opt_path")
    if not opt_path:
        raise ValueError("Не передан параметр -opt")
    
    logger = setup_logger(config.get("pipeline_logger", {}).get("log_file", "pipeline.log"))
    logger.info("🚀 СТАРТ КОНВЕЙЕРА (используется CustomNEFPairDataset)")
    
    device = get_torch_device()
    
    try:
        # 1. Генерация датасета (если нужно)
        clean_dataset_flag = config.get("clean_dataset", False)
        ensure_dataset_ready(config, clean_dataset=clean_dataset_flag)

        # 2. Создание правильного датасета для визуального контроля и smoke-теста
        dataset_root = config['path']['dataset_root']
        train_lq_dir = os.path.join(dataset_root, 'train', 'lq_inputs')
        train_hq_dir = os.path.join(dataset_root, 'train', 'hq_targets')
        
        # Проверяем, что папки существуют (если датасет сгенерирован)
        if not os.path.exists(train_lq_dir) or not os.path.exists(train_hq_dir):
            logger.error(f"Папки датасета не найдены: {train_lq_dir} или {train_hq_dir}")
            sys.exit(1)
        
        # Создаём экземпляр CustomNEFPairDataset (без аугментаций для визуализации)
        # Временно отключаем аугментации, чтобы получить детерминированный превью
        config['datasets'] = config.get('datasets', {})
        config['datasets']['train'] = config['datasets'].get('train', {})
        config['datasets']['train']['use_flip'] = False
        config['datasets']['train']['use_rot'] = False
        
        dataset = CustomNEFPairDataset(train_lq_dir, train_hq_dir, opt=config)
        logger.info(f"Датасет загружен, {len(dataset)} пар LQ/HQ")

        # 3. Визуальный контроль (сохраняет превью)
        run_visual_control(dataset, config)
        
        # 4. Smoke-тест (проверка градиентов и совместимости с GPU)
        model = create_nafnet_model(config, device)
        run_smoke_test(model, dataset, config, device)
        
        logger.info("🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
    except Exception as e:
        logger.critical(f"💥 АВАРИЙНОЕ ЗАВЕРШЕНИЕ: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()