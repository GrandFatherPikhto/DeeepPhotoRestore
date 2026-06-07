import os
import sys
import argparse
import yaml

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from libraries.logger import get_logger

logger = get_logger()

# Настройки цветов для терминала
RED = "\033[91m"
RESET = "\033[0m"
CROSS = f"{RED}✘{RESET}"

def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Tanahen Image Restoration Pipeline")
    parser.add_argument("-opt", type=str, required=True, help="Путь к конфигурационному файлу YAML")
    parser.add_argument("--clean", action="store_true", help="[DEPRECATED] Очистить всё (датасет + визуализации)")
    parser.add_argument("--clean-dataset", action="store_true", help="Очистить папки train/test датасета")
    parser.add_argument("--clean-visuals", action="store_true", help="Очистить папку с визуализациями")
    parser.add_argument("--clean-training", action="store_true", help="Очистить эксперимент (удалить папку experiments/имя)")
    parser.add_argument("--resume", type=str, default=None, help="Путь к чекпоинту для продолжения обучения")
    return parser.parse_args()

def load_yaml_config(config_path):
    """Чтение и валидация YAML-конфигурации."""
    if not os.path.exists(config_path):
        logger.error(f"{CROSS}\tКонфигурационный файл не найден: {config_path}")
        sys.exit(1)  # Завершает работу с кодом ошибки
        
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Конфигурация успешно загружена из файла: {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Ошибка синтаксиса в YAML-файле: {e}")
            raise e

def get_pipeline_config():
    args = parse_args()
    config = load_yaml_config(args.opt)
    
    # Если передан старый --clean, включаем оба старых флага (но не --clean-training)
    if args.clean:
        args.clean_dataset = True
        args.clean_visuals = True
    
    config["clean_dataset"] = args.clean_dataset
    config["clean_visuals"] = args.clean_visuals
    config["clean_training"] = args.clean_training
    config["resume"] = args.resume
    config["opt_path"] = args.opt

    exp_name = config.get('name')
    if exp_name:
        vis_cfg = config.get('visuals_logger', {})
        if 'output_dir' in vis_cfg:
            vis_cfg['output_dir'] = vis_cfg['output_dir'].replace('{name}', exp_name)
        log_cfg = config.get('pipeline_logger', {})
        if 'log_file' in log_cfg:
            log_cfg['log_file'] = log_cfg['log_file'].replace('{name}', exp_name)
    
    resume_cfg = config.get('path', {})
    if 'resume_path' in resume_cfg:
        resume_cfg['resume_path'] = resume_cfg['resume_path'].replace('{name}', exp_name)

    # === [РЕФОРМА КОНФИГУРАЦИИ: ВАРИАНТ А] ===
    # 1. Проверка на наличие устаревшего параметра (Deprecation Check)
    train_cfg = config.get('datasets', {}).get('train', {})
    if 'upscale_factor' in train_cfg:
        logger.warning(
            f"\n⚠️  [DEPRECATION WARNING]: Обнаружен устаревший параметр "
            f"'datasets.train.upscale_factor' в файле {args.opt}.\n"
            f"Данная строка полностью ПРОИГНОРИРОВАНА. Единственным легитимным "
            f"источником истины масштаба теперь является 'network_g.upscale_factor'.\n"
        )
        
    # 2. Жёсткий замок архитектурной целостности (Критерий приёмки №4)
    if 'network_g' not in config or 'upscale_factor' not in config['network_g']:
        logger.critical("💥 КРИТИЧЕСКАЯ ОШИБКА: Конфигурационный файл не содержит обязательный параметр 'network_g.upscale_factor'!")
        raise ValueError("Архитектурная целостность нарушена: 'network_g.upscale_factor' является обязательным полем ТЗ.")
        
    return config
