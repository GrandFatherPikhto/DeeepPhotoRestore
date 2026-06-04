import argparse
import os
import yaml
from libraries.pipeline_logger import get_logger

logger = get_logger()

def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Tanahen Image Restoration Pipeline")
    parser.add_argument(
        "-opt", type=str, required=True,
        help="Путь к конфигурационному файлу YAML (например, options/train/RAW_NAFNet_NikonD600.yml)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="[DEPRECATED] Очистить всё (датасет и визуализации)"
    )
    parser.add_argument(
        "--clean-dataset", action="store_true",
        help="Очистить папки train/test датасета"
    )
    parser.add_argument(
        "--clean-visuals", action="store_true",
        help="Очистить папку с визуализациями"
    )
    return parser.parse_args()

def load_yaml_config(config_path):
    """Чтение и валидация YAML-конфигурации."""
    if not os.path.exists(config_path):
        logger.error(f"Конфигурационный файл не найден по пути: {config_path}")
        raise FileNotFoundError(f"Missing config: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Конфигурация успешно загружена из файла: {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Ошибка синтаксиса в YAML-файле: {e}")
            raise e

def get_pipeline_config():
    """Точка входа модуля: возвращает объединенный словарь настроек."""
    args = parse_args()
    config = load_yaml_config(args.opt)
    
    # Если передан старый --clean, включаем оба новых флага
    if args.clean:
        args.clean_dataset = True
        args.clean_visuals = True
    
    # Сохраняем флаги в config
    config["clean_dataset"] = getattr(args, "clean_dataset", False)
    config["clean_visuals"] = getattr(args, "clean_visuals", False)
    config["opt_path"] = args.opt
    
    return config