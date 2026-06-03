## VVG19

Вот краткая шпаргалка по командам запуска и обновленный код для проверки результатов.
------------------------------
## 🧱 ШПАРГАЛКА ПО ЗАПУСКАМ ОБУЧЕНИЯ
Перед каждым запуском в терминале WSL:

cd /home/grand/Projects/Python/Tanahen/NAFNet
source ../.venv/bin/activate


* Вариант 1. Базовое обучение (Только пиксели / L1Loss)

CUDA_VISIBLE_DEVICES=0 python3 basicsr/train.py -opt options/train/Tanahen/train_NAFNet_JPEG.yml

Результат (веса): сохраняется в experiments/NAFNet-JPEG-4070Ti/models/
* Вариант 2. Продвинутое обучение (Пиксели + Восприятие / VGG19)

CUDA_VISIBLE_DEVICES=0 python3 basicsr/train.py -opt options/train/Tanahen/train_vgg19.yml

Результат (веса): сохраняется в experiments/NAFNet-JPEG-4070Ti-vgg19/models/

------------------------------
## 🔮 КАК ВЫГЛЯДИТ inference.py ТЕПЕРЬ?
Сам код нейросети (архитектура NAFNet) от добавления новых учителей не изменился — она по-прежнему принимает картинку и выдает картинку.
Но теперь у тебя есть два разных эксперимента. Чтобы запустить проверку именно на новой модели с VGG-19, тебе нужно просто изменить путь к файлу весов (MODEL_PATH).
Создай или обнови файл inference.py в корне Tanahen следующим кодом:

import osimport sysimport torchimport numpy as npfrom PIL import Image
# Авто-настройка базовой директории TanahenBASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Подключаем репозиторий NAFNet
sys.path.append(os.path.join(BASE_DIR, "NAFNet"))from basicsr.models.archs.NAFNet_arch import NAFNet  # Соблюдаем регистр букв!
# ==================== НАСТРОЙКИ ПУТЕЙ ====================
# Указываем путь к весам эксперимента с VGG19!# Если обучение еще идет, можно заменить 'net_g_latest.pth' на конкретный шаг, например 'net_g_5000.pth'MODEL_PATH = os.path.join(BASE_DIR, "NAFNet/experiments/NAFNet-JPEG-4070Ti-vgg19/models/net_g_latest.pth")
# Какую "шакальную" картинку лечитьINPUT_IMAGE = os.path.join(BASE_DIR, "dataset_ready/train/lq_inputs/ra0cc3d11t.jpg")
# Куда положить шедеврOUTPUT_IMAGE = os.path.join(BASE_DIR, "fixed_by_nafnet_vgg.png")
# =========================================================
def main():
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Ошибка: Не найден входной файл: {INPUT_IMAGE}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Ошибка: Не найден файл весов новой модели: {MODEL_PATH}")
        print("Дождись хотя бы первых чекпоинтов обучения VGG19!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Запускаем инференс VGG-модели на: {device}")

    # Инициализация точно такой же архитектуры
    model = NAFNet(img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    
    # Заливаем новые "мозги", у которых активировано человеческое восприятие
    print("🧠 Загружаем прокачанные веса VGG19...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    # Прогоняем картинку через сеть
    print("📸 Обработка кадра...")
    img_pil = Image.open(INPUT_IMAGE).convert('RGB')
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(img_tensor)

    # Сохраняем результат
    output_tensor = output_tensor.squeeze(0).clamp(0, 1).cpu()
    output_np = np.transpose(output_tensor.numpy(), (1, 2, 0))
    output_np = (output_np * 255.0).astype(np.uint8)

    fixed_img = Image.fromarray(output_np)
    fixed_img.save(OUTPUT_IMAGE)
    print(f"🎉 Готово! Результат новой модели сохранен в: {OUTPUT_IMAGE}")
if __name__ == '__main__':
    main()

## 🏎️ Запуск инференса:
Из корня папки Tanahen:

python3 inference.py

После завершения ты сможешь открыть три картинки:

   1. Исходную плохую (dataset_ready/train/lq_inputs/ra0cc3d11t.jpg)
   2. Старый результат (fixed_by_nafnet.png) — очищенный только по пикселям.
   3. Новый результат (fixed_by_nafnet_vgg.png) — очищенный с учетом восприятия.

Если хочешь, пока модель добирается до первой тысячи шагов, давай набросаем план для Варианта Б (Апскейлинг), чтобы знать, как научить ИИ увеличивать картинку в размерах! Начнем разбор?

