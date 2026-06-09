# -*- coding: utf-8 -*-
import os
import re
import sys
import yaml
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

class BlockListDumper(yaml.SafeDumper):
    """Кастомный дампер для вывода простых списков в одну строку [X, Y]."""
    def represent_sequence(self, tag, sequence, flow_style=None):
        if sequence and all(not isinstance(i, (dict, list)) for i in sequence):
            flow_style = True
        return super().represent_sequence(tag, sequence, flow_style)


def replace_placeholder(data, name_value):
    if isinstance(data, dict):
        return {k: replace_placeholder(v, name_value) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_placeholder(item, name_value) for item in data]
    elif isinstance(data, str):
        return data.replace("{name}", name_value)
    return data


def get_dsc_numbers(dataset_root_str: str):
    lq_inputs_dir = Path(dataset_root_str) / "test" / "lq_inputs"
    if not lq_inputs_dir.exists():
        print(f"Предупреждение: Директория {lq_inputs_dir} не существует!")
        return []

    file_numbers = []
    for file_path in lq_inputs_dir.glob("*"):
        if file_path.suffix.lower() in [".tiff", ".tif"]:
            match = re.search(r"\d+", file_path.name)
            if match:
                file_numbers.append(match.group())
    return sorted(file_numbers)


def get_epoch_paths(config, num_epoch):
    epoch_paths = []
    for dsc_num in config.dsc_numbers:
        epoch_path = f"{config.experiment_dir_str}/val_predictions/epoch_{num_epoch}_DSC_{dsc_num}.png"
        if not os.path.exists(epoch_path):
            raise FileNotFoundError(f"Файл {epoch_path} не найден")
        epoch_paths.append(epoch_path)
    return epoch_paths


def get_lq_paths(config):
    lq_paths = []
    for dsc_num in config.dsc_numbers:
        lq_path = f"{config.experiment_dir_str}/lq_preview/DSC_{dsc_num}_bayer.png"
        if not os.path.exists(lq_path):
            raise FileNotFoundError(f"Файл {lq_path} не найден")
        lq_paths.append(lq_path)
    return lq_paths


def get_hq_paths(config):
    hq_paths = []
    for dsc_num in config.dsc_numbers:
        hq_path = f"{config.dataset_root_str}/test/hq_targets/DSC_{dsc_num}.png"
        if not os.path.exists(hq_path):
            raise FileNotFoundError(f"Файл {hq_path} не найден")
        hq_paths.append(hq_path)
    return hq_paths


def get_valid_path(config_dict: dict, keys: list | tuple, param_name: str) -> str:
    target = config_dict
    for key in keys:
        if isinstance(target, dict):
            target = target.get(key)
        else:
            target = None
            break
    if not target or not os.path.exists(str(target)):
        raise FileNotFoundError(f"Параметр '{param_name}' некорректен или путь не существует: '{target}'")
    return str(target)


def prepare_base_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_yml = yaml.safe_load(f)

    name_value = config_yml.get("name")
    if not name_value:
        name_value = Path(config_path).stem

    prepared_yml = replace_placeholder(config_yml, name_value)

    config = SimpleNamespace()
    config.dataset_root_str = get_valid_path(prepared_yml, ["path", "dataset_root"], "dataset_root")
    config.experiment_dir_str = get_valid_path(prepared_yml, ["experiment_dir"], "experiment_dir")
    return config


def generate_report_yaml(config, output_path):
    report_data = {"pairs": []}
    report_dir = os.path.join(config.experiment_dir_str, "report")
    epochs_out_dir = os.path.join(report_dir, "epochs")
    source_out_dir = os.path.join(report_dir, "source")
    repair_out_dir = os.path.join(report_dir, "repair")   # новая папка

    os.makedirs(epochs_out_dir, exist_ok=True)
    os.makedirs(source_out_dir, exist_ok=True)
    os.makedirs(repair_out_dir, exist_ok=True)

    for i, dsc_num in enumerate(config.dsc_numbers):
        # 1. Пары Эпох
        epoch_img_name = f"epoch_{config.start_epoch}_{config.finish_epoch}_DSC_{dsc_num}.png"
        report_data["pairs"].append({
            "img1": config.start_epoch_previews[i],
            "img2": config.finish_epoch_previews[i],
            "output": os.path.join(epochs_out_dir, epoch_img_name),
            "dsc_num": dsc_num,          # добавили номер
            "type": "epochs"             # опционально
        })

        # 2. Пары Исходников (HQ vs LQ)
        source_img_name = f"hq_lq_DSC_{dsc_num}.png"
        report_data["pairs"].append({
            "img1": config.hq_targets[i],
            "img2": config.lq_previews[i],
            "output": os.path.join(source_out_dir, source_img_name),
            "dsc_num": dsc_num,
            "type": "source"
        })

        # 3. Пары Ремонта (LQ vs восстановленное)  ← НОВОЕ
        repair_img_name = f"repair_DSC_{dsc_num}.png"
        report_data["pairs"].append({
            "img1": config.lq_previews[i],                           # исходное LQ (демозаиченное)
            "img2": config.finish_epoch_previews[i],                 # восстановленное финишной эпохой
            "output": os.path.join(repair_out_dir, repair_img_name),
            "dsc_num": dsc_num,
            "type": "repair"
        })

    # Глобальные настройки (без изменений)
    report_data["image_size"] = config.image_size
    report_data["spacing"] = config.spacing
    report_data["background_color"] = config.background_color
    report_data["border"] = {
        "enabled": config.border_enabled,
        "color": config.border_color,
        "thickness": config.border_thickness,
    }
    report_data["metrics"] = {
        "enabled": getattr(config, 'metrics_enabled', True),
        "list": getattr(config, 'metrics_list', ['psnr', 'ssim']),
        "position": getattr(config, 'metrics_position', 'top_left'),
        "color": getattr(config, 'metrics_color', [255, 255, 255])
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(report_data, f, Dumper=BlockListDumper, allow_unicode=True, sort_keys=False)

    print(f"Конфигурация отчёта сохранена в: {output_path}")
    return report_data["pairs"]