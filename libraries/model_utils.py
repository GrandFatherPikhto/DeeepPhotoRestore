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
    def __init__(self, original_model, in_channels=4, out_channels=3, upscale_factor=2):
        super().__init__()
        
        self.net = original_model
        self.upscale_factor = upscale_factor
        
        # 1. Начальный блок: (B, 4, H, W) -> (B, 3, 2H, 2W)
        mid_channels = out_channels * (upscale_factor ** 2)
        self.pre_upsample = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        
        # 🎯 ИСПРАВЛЕНИЕ: Пытаемся вытащить img_channels или img_channel, 
        # а если библиотека их прячет — берём переданный in_channels (4) как безопасный фолбек
        self.net_in_channels = getattr(original_model, 'img_channels', 
                                       getattr(original_model, 'img_channel', in_channels))
        
        # 2. Адаптер каналов перед входом в NAFNet (переводим 3 канала RGB во входные каналы сети)
        self.to_nafnet_ch = nn.Conv2d(out_channels, self.net_in_channels, kernel_size=3, padding=1)
        
        # 3. Финальный блок: возвращаем выученные признаки NAFNet обратно в честный RGB
        self.post_process = nn.Conv2d(self.net_in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Шаг 1: Апскейл геометрии и сборка цвета из Bayer
        x_scaled = self.pre_upsample(x)
        
        # Шаг 2: Подгонка под входные каналы NAFNet (4 канала)
        x_naf = self.to_nafnet_ch(x_scaled)
        
        # Шаг 3: Прогон через ВСЮ глубину NAFNet в высоком разрешении
        out_features = self.net(x_naf)
        if isinstance(out_features, dict):
            out_features = out_features['out']
            
        # Шаг 4: Финальная фильтрация в RGB
        return self.post_process(out_features)

def create_nafnet_model(config, device, pretrained_path=None):
    """
    Создаёт модель NAFNet.
    Если указан pretrained_path, загружает предобученные веса,
    заменяя conv_first для работы с 4 входными каналами.
    """
    net_cfg = config.get("network_g", {})
    in_ch = net_cfg.get("num_in_ch", 4)   # 4 входных канала (RGGB)
    out_ch = net_cfg.get("num_out_ch", 3)  # 3 выходных канала (RGB)
    width = net_cfg.get("width", 32)
    middle_blk_num = net_cfg.get("middle_blk_num", 12)
    enc_blk_nums = net_cfg.get("enc_blk_nums", [2, 2, 4, 8])
    dec_blk_nums = net_cfg.get("dec_blk_nums", [2, 2, 2, 2])

    # 1. Создаём базовую сеть NAFNet с 4 входными каналами
    raw_model = NAFNet(
        img_channel=in_ch,  # Уже стоит 4
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums
    )

    # 2. Загружаем предобученные веса, если они есть
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Загрузка предобученных весов из {pretrained_path}")
        pretrained_weights = torch.load(pretrained_path, map_location='cpu')

        # Извлекаем state_dict из загруженного файла (может лежать в ключе 'params')
        pretrained_dict = pretrained_weights
        if 'params' in pretrained_weights:
            pretrained_dict = pretrained_weights['params']
        elif 'state_dict' in pretrained_weights:
            pretrained_dict = pretrained_weights['state_dict']

        # Получаем state_dict нашей новой модели (с 4 входными каналами)
        model_dict = raw_model.state_dict()

        # Загружаем только те слои, у которых размерности совпадают
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
            logger.warning(f"Пропущенные слои из-за несовпадения размерностей: {', '.join(skipped_layers)}")

        # Обновляем state_dict модели
        model_dict.update(pretrained_dict_filtered)
        raw_model.load_state_dict(model_dict, strict=False)  # strict=False позволяет пропустить отсутствующие ключи
        logger.info(f"Загружено {len(pretrained_dict_filtered)} слоёв из предобученной модели.")
    else:
        logger.info("Предобученные веса не указаны или не найдены. Обучение с нуля.")

    # 3. Оборачиваем модель в демозаик и апскейл
    upscale_factor = net_cfg.get("upscale_factor", 2)
    model = NAFNetDemosaicSuperResolutionWrapper(raw_model, in_channels=in_ch, out_channels=out_ch, upscale_factor=upscale_factor)    
    model = model.to(device)

    logger.info(f"🚀 [Архитектура] Честный демозаик NAFNet создан!")
    logger.info(f"Вход (RAW subchannels): [{in_ch} ch] -> Выход (PixelShuffle Апскейл): [{out_ch} ch RGB]")
    return model