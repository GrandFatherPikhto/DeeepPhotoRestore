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
        help="Флаг полной очистки папки с дебаг-визуализациями перед запуском"
    )
    return parser.parse_args()

def load_yaml_config(config_path):
    """Чтение и валидация YAML-конфигурации."""
    if not os.path.exists(config_path):
        logger.error(f"Конфигурационный файл не найден по пути: {config_path}")
        raise FileNotFoundError(f"Missing config: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            # Используем SafeLoader для безопасного чтения структуры
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
    
    # Инжектируем флаг --clean прямо в общий словарь конфигурации для удобства
    config["clean_visuals"] = args.clean
    return config
