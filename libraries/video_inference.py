#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import torch
from tqdm import tqdm

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from libraries.logger import get_logger

logger = get_logger()

def load_model(config, device, checkpoint_path):
    """Загружает модель NAFNet из чекпоинта."""
    from libraries.model_utils import create_nafnet_model
    model = create_nafnet_model(config, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    logger.info(f"Модель загружена из {checkpoint_path}")
    return model

def process_video(model, cap, out, config, target_size, device):
    from libraries.video_degradation import degrade_frame
    from libraries.video_utils import tensor_to_frame
    # ... цикл ...
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = degrade_frame(frame_rgb, config, target_size).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, dict):
                output = output['out']
        out_frame = tensor_to_frame(output)
        out.write(out_frame)