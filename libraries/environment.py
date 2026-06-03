import os
import sys
import argparse
import yaml
import shutil
from datetime import datetime
import torch

def init_environment():
    """Парсит аргументы, загружает YAML, делает бэкапы и инициализирует CUDA"""
    parser = argparse.ArgumentParser(description="Пайплайн обучения DeepPhotoRestore")
    parser.add_argument('-opt', type=str, required=True, help='Путь к конфигурационному файлу .yml')
    parser.add_argument('--resume', type=str, default=None, help='Путь к файлу чекпоинта')
    parser.add_argument('--clean', action='store_true', help='Бэкап и полная очистка текущего эксперимента')
    args = parser.parse_args()

    # Загрузка настроек
    with open(args.opt, 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)

    exp_name = opt.get('name', 'default_exp')
    save_dir = os.path.join('experiments', exp_name)

    # Логика бэкапа и очистки
    if args.clean and os.path.exists(save_dir):
        print(f"\n🧹 [Очистка] Запущен сброс эксперимента '{exp_name}' перед обучением...")
        os.makedirs('backups', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_zip = os.path.join('backups', f"backup_train_{exp_name}_{timestamp}")
        
        print(f"  - Архивирую старые результаты в: {backup_zip}.zip")
        shutil.make_archive(backup_zip, 'zip', save_dir)
        
        print(f"  - Очищаю рабочую папку эксперимента: {save_dir}")
        shutil.rmtree(save_dir)
        
        # Удаляем старый дефолтный файл возобновления, если он есть
        resume_path_yml = opt.get('path', {}).get('resume_path', 'checkpoints/resume_simple_jpeg.pth')
        if os.path.exists(resume_path_yml):
            print(f"  - Удаляю старый файл возобновления: {resume_path_yml}")
            try: os.remove(resume_path_yml)
            except Exception: pass

    os.makedirs(save_dir, exist_ok=True)
    
    # Инициализация девайса
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return args, opt, save_dir, device
