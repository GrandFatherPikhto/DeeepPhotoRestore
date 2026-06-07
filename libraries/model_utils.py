import os
import sys
import torch
import torch.nn as nn

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from basicsr.models.archs.NAFNet_arch import NAFNet
from libraries.logger import get_logger

logger = get_logger()


class NAFNetDemosaicSuperResolutionWrapper(nn.Module):
    """
    Правильная обёртка для совместной демозаики и суперразрешения.
    NAFNet работает в низком разрешении на 4 каналах (RGGB).
    После сети – свёртка 4 -> (out_ch * scale²) и PixelShuffle.
    """
    def __init__(self, original_model, in_channels=4, out_channels=3, upscale_factor=4):
        super().__init__()
        self.net = original_model
        self.upscale_factor = upscale_factor

        # Количество промежуточных каналов для PixelShuffle
        mid_channels = out_channels * (upscale_factor ** 2)   # 3 * 16 = 48

        # Блок апскейла: conv + PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        # x: (B, 4, H, W) – низкое разрешение (например, 128×128)
        feat = self.net(x)           # (B, 4, H, W) – сеть выдаёт 4 канала
        out = self.upsample(feat)    # (B, 3, H*scale, W*scale)
        return out


def create_nafnet_model(config, device, pretrained_path=None):
    """
    Создаёт модель NAFNet для JDSR.
    - Вход: 4 канала (RGGB)
    - Выход: 3 канала (RGB) с масштабом upscale_factor
    """
    net_cfg = config.get("network_g", {})
    in_ch = net_cfg.get("num_in_ch", 4)        # 4
    out_ch = net_cfg.get("num_out_ch", 3)      # 3
    width = net_cfg.get("width", 32)
    middle_blk_num = net_cfg.get("middle_blk_num", 12)
    enc_blk_nums = net_cfg.get("enc_blk_nums", [2, 2, 4, 8])
    dec_blk_nums = net_cfg.get("dec_blk_nums", [2, 2, 2, 2])
    upscale_factor = net_cfg.get("upscale_factor", 4)

    # 1. Базовый NAFNet: принимает и отдаёт 4 канала, работает в низком разрешении
    raw_model = NAFNet(
        img_channel=in_ch,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums
    )

    # 2. Загрузка предобученных весов (если есть)
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Загрузка предобученных весов из {pretrained_path}")
        pretrained_weights = torch.load(pretrained_path, map_location='cpu')

        # Извлекаем state_dict
        pretrained_dict = pretrained_weights
        if 'params' in pretrained_weights:
            pretrained_dict = pretrained_weights['params']
        elif 'state_dict' in pretrained_weights:
            pretrained_dict = pretrained_weights['state_dict']

        model_dict = raw_model.state_dict()

        # Фильтруем совместимые слои
        pretrained_dict_filtered = {}
        skipped_layers = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict_filtered[k] = v
                else:
                    skipped_layers.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
            else:
                skipped_layers.append(f"{k} (not in model)")

        if skipped_layers:
            logger.warning(f"Пропущенные слои: {', '.join(skipped_layers)}")

        model_dict.update(pretrained_dict_filtered)
        raw_model.load_state_dict(model_dict, strict=False)
        logger.info(f"Загружено {len(pretrained_dict_filtered)} слоёв из предобученной модели.")
    else:
        logger.info("Предобученные веса не указаны. Обучение с нуля.")

    # 3. Оборачиваем модель в правильную обёртку
    model = NAFNetDemosaicSuperResolutionWrapper(
        raw_model,
        in_channels=in_ch,
        out_channels=out_ch,
        upscale_factor=upscale_factor
    )
    model = model.to(device)

    logger.info(f"🚀 Модель JDSR создана: вход {in_ch} каналов (RGGB), "
                f"выход {out_ch} каналов (RGB), масштаб {upscale_factor}")
    return model