import torch
import torch.nn as nn
from basicsr.models.archs.NAFNet_arch import NAFNet
from libraries.logger import get_logger

logger = get_logger()

class NAFNetRGBWrapper(nn.Module):
    """Обёртка для изменения числа выходных каналов NAFNet."""
    def __init__(self, original_model, nafnet_channels, out_channels):
        super().__init__()
        self.net = original_model
        if nafnet_channels != out_channels:
            self.post_process = nn.Conv2d(nafnet_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.post_process = nn.Identity()
    def forward(self, x):
        out = self.net(x)
        return self.post_process(out)

def create_nafnet_model(config, device):
    net_cfg = config.get("network_g", {})
    in_ch = net_cfg.get("num_in_ch", 4)   # входных каналов (RGGB)
    out_ch = net_cfg.get("num_out_ch", 3)  # выходных (RGB)
    width = net_cfg.get("width", 32)
    middle_blk_num = net_cfg.get("middle_blk_num", 12)
    enc_blk_nums = net_cfg.get("enc_blk_nums", [2, 2, 4, 8])
    dec_blk_nums = net_cfg.get("dec_blk_nums", [2, 2, 2, 2])
    
    raw_model = NAFNet(
        img_channel=in_ch,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums
    )
    model = NAFNetRGBWrapper(raw_model, nafnet_channels=in_ch, out_channels=out_ch)
    model = model.to(device)
    logger.info(f"Модель NAFNet создана: вход={in_ch}, выход={out_ch}, ширина={width}")
    return model