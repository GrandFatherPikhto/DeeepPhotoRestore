#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from libraries.config import load_yaml_config
from libraries.device import get_torch_device
from libraries.logger import setup_logger, get_logger
from libraries.video_utils import open_video, create_video_writer
from libraries.video_degradation import degrade_frame
from libraries.video_inference import load_model, process_video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', required=True, help='Путь к конфигурационному YAML (например, options/video/process_video.yml)')
    parser.add_argument('--input', required=True, help='Путь к входному видео (чистое RGB)')
    parser.add_argument('--output', required=True, help='Путь для сохранения восстановленного видео')
    args = parser.parse_args()

    config = load_yaml_config(args.opt)
    log_cfg = config.get('pipeline_logger', {})
    log_file = log_cfg.get('log_file', 'video_processing.log')
    setup_logger(log_file)
    logger = get_logger()

    device = get_torch_device()
    logger.info(f"Устройство: {device}")

    model = load_model(config, device, config['path']['checkpoint'])
    cap, fps, width, height, total = open_video(args.input)
    out = create_video_writer(args.output, fps, width, height, codec=config.get('output', {}).get('codec', 'mp4v'))

    target_size = config.get('gt_size', 256)
    process_video(model, cap, out, config, target_size, device)

    cap.release()
    out.release()
    logger.info(f"Видео сохранено в {args.output}")

if __name__ == '__main__':
    main()