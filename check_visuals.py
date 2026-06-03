#!./.venv/bin/python3
import os
import numpy as np
import tifffile
from PIL import Image

def main():
    # Пути к сгенерированному датасету
    dataset_root = "datasets/nef_nafnet"
    lq_dir = os.path.join(dataset_root, "train/lq_inputs")
    gt_dir = os.path.join(dataset_root, "train/hq_targets")
    
    if not os.path.exists(lq_dir) or not os.listdir(lq_dir):
        print("❌ Папка сгенерированного датасета пуста или не найдена!")
        return

    # Берем самый первый файл для проверки
    tiff_files = [f for f in os.listdir(lq_dir) if f.endswith('.tiff')]
    if not tiff_files:
        print("❌ В папке lq_inputs нет .tiff файлов!")
        return
        
    test_file = tiff_files[0]
    base_name = test_file.replace('_bayer.tiff', '')
    print(f"🔍 Запуск визуального контроля для кадра: {base_name}")

    # 1. Читаем сгенерированный LQ 4-канальный TIFF
    lq_packed = tifffile.imread(os.path.join(lq_dir, test_file))
    print(f"   • Размерность LQ на диске: {lq_packed.shape} (Должно быть [H, W, 4])")
    
    # Извлекаем RGGB каналы (они в диапазоне 0-65535)
    r  = lq_packed[:, :, 0].astype(np.float32) / 65535.0
    g1 = lq_packed[:, :, 1].astype(np.float32) / 65535.0
    g2 = lq_packed[:, :, 2].astype(np.float32) / 65535.0
    b  = lq_packed[:, :, 3].astype(np.float32) / 65535.0
    
    # 2. Собираем быстрое превью (усредняем зеленые каналы, склеиваем в RGB)
    g_mean = (g1 + g2) / 2.0
    preview_rgb = np.stack([r, g_mean, b], axis=2)
    preview_rgb = np.clip(preview_rgb * 255.0, 0, 255).astype(np.uint8)

    # 3. Загружаем эталонный GT PNG
    gt_img = Image.open(os.path.join(gt_dir, f"{base_name}.png")).convert('RGB')
    
    # 4. Сохраняем тестовую пару рядом в корень для визуального сравнения
    preview_out_path = "debug_lq_preview.png"
    gt_out_path = "debug_gt_reference.png"
    
    Image.fromarray(preview_rgb).save(preview_out_path)
    gt_img.save(gt_out_path)
    
    print("\n📸 === ВИЗУАЛЬНЫЙ ТЕСТ ЗАВЕРШЕН ===")
    print(f"   1. Восстановленное превью LQ (с шумом и даунскейлом) сохранено в: {preview_out_path}")
    print(f"   2. Исходный чистый референс GT сохранен в: {gt_out_path}")
    print("===================================\n")
    print("💡 Откройте эти два файла глазами. На картинке preview_out_path вы должны увидеть")
    print("чёткие контуры вашей реальной сцены (уменьшенные в 4-8 раз) с наложенным цифровым зерном.")
    print("Если картинка узнаваема и цвета на месте — конвейер работает идеально!")

if __name__ == "__main__":
    main()
