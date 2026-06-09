#!.venv/bin/python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import libraries.report_config as cm
import libraries.report_generator as ip
import libraries.lq_generator as lg

sys.path.insert(0, str(Path(__file__).parent))


def parse_int_list(string_value):
    try:
        return [int(x.strip()) for x in string_value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Список должен состоять из чисел через запятую (например: 512,512)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Конвейер: генерация конфига + склейка пар изображений"
    )
    parser.add_argument(
        "-opt", required=True, help="Путь к YAML конфигу обучения"
    )
    parser.add_argument("--start_epoch", default="0", help="Стартовая эпоха")
    parser.add_argument("--finish_epoch", required=True, help="Конечная эпоха")
    parser.add_argument(
        "--output", default="./report.yml", help="Файл для сохранения YAML-отчёта"
    )

    # ИСПРАВЛЕНО: Дефолты заданы строками, parse_int_list отработает корректно!
    parser.add_argument(
        "--image_size",
        type=parse_int_list,
        default="512,512",
        help="Размер изображений (W,H)",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=10,
        help="Зазор между картинками в px",
    )
    parser.add_argument(
        "--background_color",
        type=parse_int_list,
        default="255,255,255",
        help="Цвет фона (R,G,B)",
    )
    parser.add_argument(
        "--border_disabled",
        action="store_false",
        dest="border_enabled",
        help="Отключить рамки",
    )
    parser.add_argument(
        "--border_color",
        type=parse_int_list,
        default="0,0,255",
        help="Цвет рамки (R,G,B)",
    )
    parser.add_argument(
        "--border_thickness",
        type=int,
        default=1,
        help="Толщина рамки в px",
    )

    # НАСТРОЙКИ МЕТРИК
    parser.add_argument("--metrics_disabled", action="store_false", dest="metrics_enabled", help="Отключить вывод метрик")
    parser.add_argument("--metrics_list", type=lambda s: [x.strip().lower() for x in s.split(",")], default="psnr,ssim", help="Какие метрики считать через запятую (psnr,ssim)")
    parser.add_argument("--metrics_position", type=str, default="top_left", choices=["top_left", "top_right", "bottom_left", "bottom_right"], help="Расположение плашки с метриками")
    parser.add_argument("--metrics_color", type=parse_int_list, default="255,255,255", help="Цвет шрифта метрик (R,G,B)")    

    args = parser.parse_args()

    config = cm.prepare_base_config(args.opt)

    config.start_epoch = args.start_epoch
    config.finish_epoch = args.finish_epoch
    config.dsc_numbers = cm.get_dsc_numbers(config.dataset_root_str)

    config.image_size = args.image_size
    config.spacing = args.spacing
    config.background_color = args.background_color
    config.border_enabled = args.border_enabled
    config.border_color = args.border_color
    config.border_thickness = args.border_thickness

    # Пробрасываем настройки метрик
    config.metrics_enabled = args.metrics_enabled
    config.metrics_list = args.metrics_list
    config.metrics_position = args.metrics_position
    config.metrics_color = args.metrics_color    

    if not config.dsc_numbers:
        print("Ошибка: Не найдено ни одного номера файла для обработки.")
        return

    # 3. Автоматически генерируем превью из raw-tiff
    lg.generate_previews_from_config(config)

    config.start_epoch_previews = cm.get_epoch_paths(config, config.start_epoch)
    config.finish_epoch_previews = cm.get_epoch_paths(
        config, config.finish_epoch
    )
    config.lq_previews = cm.get_lq_paths(config)
    config.hq_targets = cm.get_hq_paths(config)

    pairs_list = cm.generate_report_yaml(config, args.output)

    print("\nНачинаем процесс склейки изображений...")

    # Создаём отображение номера DSC -> путь к HQ для быстрого доступа
    dsc_to_hq = {dsc_num: hq_path for dsc_num, hq_path in zip(config.dsc_numbers, config.hq_targets)}

    for pair in pairs_list:
        img1 = pair["img1"]
        img2 = pair["img2"]
        out = pair["output"]
        dsc_num = pair.get("dsc_num")
        if dsc_num is not None:
            hq_path = dsc_to_hq.get(dsc_num)
        else:
            # fallback для старых конфигов (без dsc_num)
            hq_path = None
            print(f"Предупреждение: в паре {pair} нет dsc_num, метрики не будут рассчитаны.")
        ip.process_pair(img1, img2, out, config, hq_target_path=hq_path)

    print("\nВсе задачи успешно выполнены!")

if __name__ == "__main__":
    main()