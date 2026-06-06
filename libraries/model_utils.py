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
    Академическая обёртка модели для честной демозаики.
    Принимает 4-канальный subchannels-пакет (H/2, W/2, 4) и с помощью 
    субпиксельного сдвига (PixelShuffle) перестраивает его в RGB (H, W, 3).
    """
    def __init__(self, original_model, in_channels=4, out_channels=3, upscale_factor=2):
        super().__init__()
        self.net = original_model
        
        # Вычисляем промежуточные каналы для PixelShuffle: 3 * (2^2) = 12 каналов
        mid_channels = out_channels * (upscale_factor ** 2)
        
        # Финальный блок восстановления пространственной сетки
        self.upsample_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        # Проход через базовые блоки NAFNet в латентном пространстве низкого разрешения
        out = self.net(x)
        # Честный апскейл разрешения в 2 раза с декомпозицией в RGB
        return self.upsample_block(out)

def create_nafnet_model(config, device):
    net_cfg = config.get("network_g", {})
    in_ch = net_cfg.get("num_in_ch", 4)   # 4 входных канала (RGGB)
    out_ch = net_cfg.get("num_out_ch", 3)  # 3 выходных канала (RGB)
    width = net_cfg.get("width", 32)
    middle_blk_num = net_cfg.get("middle_blk_num", 12)
    enc_blk_nums = net_cfg.get("enc_blk_nums", [2, 2, 4, 8])
    dec_blk_nums = net_cfg.get("dec_blk_nums", [2, 2, 2, 2])
    
    # КРИТИЧЕСКИ ДЛЯ ДЕМОЗАИКИ:
    # Базовая сеть NAFNet теперь работает на уровне латентных каналов RAW матрицы.
    # Входной и внутренний выходной размер каналов должен совпадать (4 канала).
    raw_model = NAFNet(
        img_channel=in_ch,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums
    )
    
    # Обертываем модель в честный демозаик с PixelShuffle
    model = NAFNetDemosaicSuperResolutionWrapper(raw_model, in_channels=in_ch, out_channels=out_ch)
    model = model.to(device)
    
    logger.info(f"🚀 [Архитектура] Честный демозаик NAFNet создан!")
    logger.info(f"Вход (RAW subchannels): [{in_ch} ch] -> Выход (PixelShuffle Апскейл): [{out_ch} ch RGB]")
    return model
