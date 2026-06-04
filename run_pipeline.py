#!./.venv/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
from libraries.pipeline_config import get_pipeline_config
from libraries.pipeline_logger import setup_logger

def main():
    config = get_pipeline_config()
    vis_cfg = config.get("pipeline_logger", {})
    log_path = vis_cfg.get("log_file", "pipeline.log")
    logger = setup_logger(log_path)
    
    logger.info("==================================================")
    logger.info(f"🚀 СТАРТ КОНВЕЙЕРА. ЛОГ ИДЕТ В: {log_path}")
    logger.info("==================================================")

    try:
        dataset_root = config.get("dataset_root", "datasets/nef_nafnet")
        lq_dir = os.path.join(dataset_root, "train", "lq_inputs")
        
        # Полная очистка датасета по флагу --clean
        if config.get("clean_visuals", False) and os.path.exists(dataset_root):
            logger.info(f"🧹 Передан флаг --clean. Полностью удаляем старый датасет: {dataset_root}")
            shutil.rmtree(dataset_root)
        
        # Проверяем наличие готовых файлов
        is_empty = True
        if os.path.exists(lq_dir) and len(os.listdir(lq_dir)) > 0:
            is_empty = False
        
        if is_empty:
            logger.info("📁 Папка датасета пуста или удалена. Запуск тяжелой нарезки патчей...")
            
            # Находим путь к yml из переданных аргументов командной строки
            opt_path = "options/train/RAW_NAFNet_NikonD600.yml"
            if "-opt" in sys.argv:
                opt_path = sys.argv[sys.argv.index("-opt") + 1]
            
            # Вызываем оригинальный скрипт как независимый процесс
            # sys.executable жестко гарантирует использование нашего локального .venv
            subprocess.run([sys.executable, "prepare_dataset.py", "-opt", opt_path], check=True)
            logger.info("✅ Генерация физических файлов патчей успешно завершена.")
        else:
            files_count = len(os.listdir(lq_dir))
            logger.info(f"📁 Обнаружен готовый датасет на диске ({files_count} патчей). Нарезка пропущена.")

        # Ленивый импорт остальных библиотек конвейера
        from libraries.pipeline_data import RestorationDataset
        from libraries.pipeline_visuals import run_visual_control
        from libraries.pipeline_smoke import run_smoke_test
        
        # Шаг загрузки данных, визуального контроля и smoke-теста
        dataset = RestorationDataset(config, is_train=True)
        run_visual_control(dataset, config)
        
        logger.info("Инициализация архитектуры нейросети NAFNet...")
        from basicsr.models.archs.NAFNet_arch import NAFNet
        import inspect  # Импортируем инспектор сигнатур
        
        net_cfg = config.get("network_g", {})
        
        # Инспектируем конструктор NAFNet и вытаскиваем только те имена аргументов, которые он реально ждет
        nafnet_args = inspect.signature(NAFNet.__init__).parameters.keys()
        
        # Фильтруем словарь: оставляем только валидные ключи (width, enc_blk_nums и т.д.)
        filtered_cfg = {k: v for k, v in net_cfg.items() if k in nafnet_args}
        
        # Создаем модель без риска получить TypeError из-за лишних ключей YAML
        model = NAFNet(**filtered_cfg)
        logger.info("Экземпляр модели NAFNet успешно создан в памяти.")
        
        # Тест градиентов на видеокарте
        test_sample = dataset
        run_smoke_test(model, test_sample, config)
        
        logger.info("==================================================")
        logger.info("🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО! СИСТЕМА ГОТОВА.")
        logger.info("==================================================")

    except Exception as e:
        logger.critical(f"💥 АВАРИЙНОЕ ЗАВЕРШЕНИЕ КОНВЕЙЕРА: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
