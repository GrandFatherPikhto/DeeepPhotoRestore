import os
import sys
import importlib.util
import torch
import torch.nn as nn

class NAFNetRGBWrapper(nn.Module):
    """
    Специальная обертка над NAFNet.
    Если на входе 4 канала, а на выходе нужны 3, этот класс добавляет
    финальную легкую свертку, переводящую RGGB-выход NAFNet в чистый RGB.
    """
    def __init__(self, original_model, out_channels=3):
        super().__init__()
        self.net = original_model
        # Извлекаем количество каналов, которое NAFNet выдает по умолчанию (равно входным)
        nafnet_out_channels = original_model.intro.in_channels
        
        # Если выходные каналы не совпадают, добавляем корректирующий слой
        if nafnet_out_channels != out_channels:
            self.post_process = nn.Conv2d(nafnet_out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.post_process = nn.Identity()

    def forward(self, x):
        out = self.net(x)
        return self.post_process(out)


def load_network(opt, device):
    """
    Универсальная функция загрузки и адаптации моделей (Paired JPEG / RAW NEF)
    """
    net_opt = opt.get('network_g', {})
    model_type = net_opt.get('type', 'fcn_resnet50')
    in_channels = net_opt.get('num_in_ch', 3)
    out_channels = net_opt.get('num_out_ch', 3)

    # Вариант 1: Загрузка внешней архитектуры NAFNet
    if 'arch_file' in net_opt and net_opt['arch_file']:
        arch_file_path = net_opt['arch_file']
        class_name = net_opt.get('class_name', model_type)
        
        print(f"🧠 [Фабрика] Загружаем внешнюю модель {class_name} из файла: {arch_file_path}...")
        if not os.path.exists(arch_file_path):
            raise FileNotFoundError(f"Файл архитектуры модели не найден по пути: {arch_file_path}")
            
        nafnet_root = os.path.abspath(os.path.join("modules", "NAFNet"))
        if nafnet_root not in sys.path:
            sys.path.append(nafnet_root)
            
        spec = importlib.util.spec_from_file_location("custom_arch", arch_file_path)
        custom_arch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_arch_module)
        
        ModelClass = getattr(custom_arch_module, class_name)
        
        # Инициализируем оригинальную NAFNet (на входе и выходе у нее будет по 4 канала)
        raw_model = ModelClass(
            img_channel=in_channels,
            width=net_opt.get('width', 32),
            enc_blk_nums=net_opt.get('enc_blk_nums',),
            middle_blk_num=net_opt.get('middle_blk_num', 12),
            dec_blk_nums=net_opt.get('dec_blk_nums',)
        )
        
        # Оборачиваем её в наш класс адаптации каналов (для JPEG останется Identity, для RAW сделает 4 -> 3)
        model = NAFNetRGBWrapper(raw_model, out_channels=out_channels).to(device)
        
    # Вариант 2: Загрузка проверочной встроенной модели
    else:
        print("🧠 [Фабрика] Параметр arch_file не задан. Используем встроенную модель fcn_resnet50...")
        from torchvision.models.segmentation import fcn_resnet50
        model = fcn_resnet50(num_classes=out_channels).to(device)
        if in_channels == 4:
            print("⚙️ [Фабрика] Перестраиваем первый слой fcn_resnet50 на 4 канала...")
            model.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)

    return model
