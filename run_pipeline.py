#!./.venv/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess

from libraries.config import get_pipeline_config
from libraries.logger import setup_logger
from libraries.device import get_torch_device
# from libraries.pipeline_dataset import ensure_dataset_ready
from libraries.pipeline_prepare import ensure_dataset_ready
# from libraries.pipeline_dataset import create_restoration_dataset
from libraries.pipeline_data import create_restoration_dataset
from libraries.model_utils import create_nafnet_model
from libraries.pipeline_visuals import run_visual_control
from libraries.pipeline_smoke import run_smoke_test

def main():
    config = get_pipeline_config()
    opt_path = config.get("opt_path")
    if not opt_path:
        raise ValueError("Не передан параметр -opt")
    
    logger = setup_logger(config.get("pipeline_logger", {}).get("log_file", "pipeline.log"))
    logger.info("🚀 СТАРТ КОНВЕЙЕРА")
    
    device = get_torch_device()
    
    try:
        clean_dataset_flag = config.get("clean_dataset", False)
        # ensure_dataset_ready(config, opt_path, clean_dataset=clean_dataset_flag)
        ensure_dataset_ready(config, clean_dataset=clean_dataset_flag)

        dataset = create_restoration_dataset(config, is_train=True)

        run_visual_control(dataset, config)
        
        model = create_nafnet_model(config, device)
        run_smoke_test(model, dataset, config, device)
        
        logger.info("🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
    except Exception as e:
        logger.critical(f"💥 АВАРИЙНОЕ ЗАВЕРШЕНИЕ: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
