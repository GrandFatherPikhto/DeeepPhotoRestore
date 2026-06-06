import logging
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logger(log_file="pipeline.log"):
    """
    Инициализирует двухпоточную систему логирования.
    При повторном вызове перенастраивает вывод в новый файл.
    """
    logger = logging.getLogger("TanahenPipeline")
    logger.setLevel(logging.INFO)

    # Удаляем старые обработчики, чтобы перенастроить вывод
    if logger.handlers:
        logger.handlers.clear()

    log_format = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Файловый обработчик
    import os
    log_dir = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger

def get_logger():
    """Возвращает инициализированный логгер для вызовов внутри модулей."""
    return logging.getLogger("TanahenPipeline")
