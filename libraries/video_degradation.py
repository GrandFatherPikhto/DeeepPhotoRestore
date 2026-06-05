#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision.transforms.functional as TF
from libraries.pipeline_generation_core import generate_lq_from_hq

def degrade_frame(frame_rgb, config, target_size):
    hq = frame_rgb.astype(np.float32) / 255.0
    lq_packed, _ = generate_lq_from_hq(hq, config)   # config содержит process_data
    h, w = lq_packed.shape[:2]
    if h != target_size or w != target_size:
        lq_packed = TF.resize(
            torch.from_numpy(lq_packed.transpose(2,0,1)),
            (target_size, target_size)
        ).numpy().transpose(1,2,0)
    return torch.from_numpy(lq_packed.transpose(2,0,1)).float().unsqueeze(0)

