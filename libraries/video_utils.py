#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

def open_video(input_path):
    """Открывает видеофайл, возвращает VideoCapture и параметры."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, total_frames

def create_video_writer(output_path, fps, width, height):
    """Создаёт VideoWriter для записи результата."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return out

def frame_to_tensor(frame_rgb):
    """Преобразует кадр (H,W,3) uint8 в тензор [1,3,H,W] float [0,1]."""
    img = frame_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)
    return tensor

def tensor_to_frame(tensor):
    """Преобразует выходной тензор модели [1,3,H,W] в BGR-кадр uint8."""
    out_np = tensor.squeeze(0).cpu().clamp(0,1).numpy().transpose(1,2,0)
    out_np = (out_np * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return out_bgr