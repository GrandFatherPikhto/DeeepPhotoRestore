# -*- coding: utf-8 -*-
import inspect
from basicsr.models.archs.NAFNet_arch import NAFNet
from libraries.pipeline_logger import get_logger

logger = get_logger()

def create_nafnet_model(config, device):
    """Создаёт NAFNet, фильтруя параметры из config, и перемещает на device."""
    net_cfg = config.get("network_g", {})
    # Инспектируем конструктор NAFNet
    nafnet_args = inspect.signature(NAFNet.__init__).parameters.keys()
    filtered_cfg = {k: v for k, v in net_cfg.items() if k in nafnet_args}
    model = NAFNet(**filtered_cfg)
    logger.info("Экземпляр модели NAFNet успешно создан в памяти.")
    model = model.to(device)
    logger.info(f"Модель перенесена на устройство: {device}")
    return model