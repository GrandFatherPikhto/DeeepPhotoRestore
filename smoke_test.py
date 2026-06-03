#!./.venv/bin/python3
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Импортируем наши компоненты
from libraries.model_loader import load_network
from libraries.dataset_nef import CustomNEFPairDataset
from libraries.losses import FocalFrequencyLoss

def main():
    config_path = "options/train/RAW_NAFNet_NikonD600.yml"
    if not os.path.exists(config_path):
        print(f"❌ Конфиг не найден по пути: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Тест на устройстве: {device}")

    # 1. ТЕСТ ДАТАСЕТА И ДИАПАЗОНОВ ЧИСЕЛ
    print("\n🔄 1. Тестирование загрузки данных...")
    dataset_root = opt['path']['dataset_root']
    
    train_lq_dir = os.path.join(dataset_root, "train/lq_inputs")
    train_hq_dir = os.path.join(dataset_root, "train/hq_targets")
    
    if not os.path.exists(train_lq_dir) or not os.listdir(train_lq_dir):
        print(f"❌ Ошибка: Папка {train_lq_dir} пуста или не существует. Сначала запустите prepare_dataset.py!")
        return

    try:
        dataset = CustomNEFPairDataset(
            dataroot_lq=train_lq_dir,
            dataroot_gt=train_hq_dir
        )
        # 🎯 ИСПРАВЛЕНО: Безопасное извлечение самого первого тренировочного патча по индексу [0]
        lq_tensor, gt_tensor = dataset[0]
    except Exception as e:
        print(f"❌ Ошибка при чтении элемента из датасета: {e}")
        return

    print(f"   • Вход LQ (4-канальный RAW) shape: {lq_tensor.shape} | Мин: {lq_tensor.min().item():.4f} | Макс: {lq_tensor.max().item():.4f}")
    print(f"   • Выход GT (3-канальный RGB) shape: {gt_tensor.shape} | Мин: {gt_tensor.min().item():.4f} | Макс: {gt_tensor.max().item():.4f}")

    # КРИТИЧЕСКИЙ ТЕСТ НОРМАЛИЗАЦИИ ЧИСЕЛ
    if lq_tensor.max().item() < 0.05:
        print("⚠️  БАГ ДИАПАЗОНА: Входные значения LQ слишком малы! Модель ослепнет и выдаст плоский квадрат.")
    elif lq_tensor.max().item() > 1.0 or gt_tensor.max().item() > 1.0:
        print("⚠️  БАГ ДИАПАЗОНА: Данные выходят за пределы [0.0, 1.0]! Градиенты могут взорваться.")
    else:
        print("✅ Диапазоны чисел в полном порядке (строго от 0.0 до 1.0).")

    # 2. ТЕСТ ПРЯМОГО ПРОХОДА (FORWARD)
    print("\n🧠 2. Тестирование прямого прохода архитектуры NAFNet...")
    try:
        model = load_network(opt, device)
        # Имитируем батч из 2-х картинок [B=2, C, H, W]
        inputs = torch.stack([lq_tensor, lq_tensor]).to(device)
        targets = torch.stack([gt_tensor, gt_tensor]).to(device)
        
        outputs = model(inputs)
        if isinstance(outputs, dict): outputs = outputs['out']
        print(f"   • Модель успешно выполнила свёртки. Выходной тензор shape: {outputs.shape}")
    except Exception as e:
        print(f"❌ Ошибка в графе вычислений модели (проверьте адаптер каналов враппера): {e}")
        return

    # 3. ТЕСТ ОПТИМИЗАЦИИ И ФУНКЦИЙ ПОТЕРЬ (BACKWARD)
    print("\n📉 3. Тестирование гибридного лосса (L1 + FFT Loss)...")
    try:
        criterion_l1 = nn.L1Loss()
        criterion_fft = FocalFrequencyLoss(loss_weight=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        optimizer.zero_grad()
        loss_l1 = criterion_l1(outputs, targets)
        loss_fft = criterion_fft(outputs, targets)
        total_loss = loss_l1 + loss_fft
        
        print(f"   • Пространственный L1 Loss : {loss_l1.item():.4f}")
        print(f"   • Частотный FFT Loss         : {loss_fft.item():.4f}")
        print(f"   • Итоговая сумма потерь      : {total_loss.item():.4f}")
        
        total_loss.backward()
        optimizer.step()
        print("✅ Обратный шаг градиента успешно выполнен. Ошибок компиляции графа нет!")
    except Exception as e:
        print(f"❌ Ошибка при расчете функций потерь или backward-шаге: {e}")
        return

    print("\n🚀 === СИНХРОНИЗАЦИЯ ПРОЙДЕНА УСПЕШНО! ВСЕ КОМПОНЕНТЫ ГОТОВЫ К ОБУЧЕНИЮ ===")

if __name__ == "__main__":
    main()
