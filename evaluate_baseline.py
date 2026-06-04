#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from libraries.pipeline_config import get_pipeline_config
from libraries.evaluate_baseline import run_baseline_evaluation
from libraries.pipeline_logger import setup_logger, get_logger

def main():
    config = get_pipeline_config()
    # Инициализируем логгер (если ещё нет)
    log_cfg = config.get('pipeline_logger', {})
    log_file = log_cfg.get('log_file', 'pipeline.log')
    setup_logger(log_file)   # гарантируем, что логгер настроен
    logger = get_logger()
    
    logger.info("=== Запуск оценки baseline (Bilinear, MHC) ===")
    run_baseline_evaluation(config)
    logger.info("=== Оценка завершена ===")

if __name__ == '__main__':
    main()