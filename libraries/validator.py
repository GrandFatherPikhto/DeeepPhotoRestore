import os
import torch
import cv2
import numpy as np
import tifffile
from PIL import Image
import torchvision.transforms as T

class VisualValidator:
    def __init__(self, opt):
        self.opt = opt
        self.in_channels = opt['network_g']['num_in_ch']
        exp_name = opt.get('name', 'default_experiment')
        
        # Автоматически строим пути на основе dataset_root (без повторов!)
        dataset_root = opt['path']['dataset_root']
        self.lq_val_dir = os.path.join(dataset_root, "test", "lq_inputs")
        
        # Папка, куда будут сохраняться промежуточные картинки по эпохам
        self.out_val_dir = os.path.join('experiments', exp_name, 'val_predictions')
        os.makedirs(self.out_val_dir, exist_ok=True)
        
        # Расширения файлов в зависимости от режима (JPEG или NEF)
        self.ext = '.jpg' if self.in_channels == 3 else '.tiff'

    def run_validation(self, model, epoch, device):
        """Прогоняет все тестовые файлы через модель в режиме eval и сохраняет результат"""
        if not os.path.exists(self.lq_val_dir):
            print(f"⚠️ [Валидатор] Папка {self.lq_val_dir} не найдена. Пропускаем.")
            return
            
        # Находим все файлы для теста
        files = [f for f in os.listdir(self.lq_val_dir) if f.lower().endswith(self.ext)]
        if not files:
            print(f"⚠️ [Валидатор] Нет файлов с расширением {self.ext} в {self.lq_val_dir}")
            return
            
        # Переводим модель в режим оценки (выключает Dropout/BatchNorm)
        model.eval()
        
        # Отключаем расчет градиентов ради скорости и памяти
        with torch.no_grad():
            for file_name in files:
                file_path = os.path.join(self.lq_val_dir, file_name)
                base_name = os.path.splitext(file_name)[0].replace('_bayer', '')
                
                # Подготовка входного тензора в зависимости от режима
                if self.in_channels == 3:
                    # Режим JPEG
                    input_img = Image.open(file_path).convert('RGB')
                    orig_w, orig_h = input_img.size
                    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
                    input_tensor = transform(input_img).unsqueeze(0).to(device)
                else:
                    # Режим NEF / RAW
                    bayer = tifffile.imread(file_path).astype(np.float32) / 65535.0
                    orig_h, orig_w = bayer.shape
                    
                    # 🎯 ИСПРАВЛЕНИЕ: Гарантируем четность размеров перед нарезкой каналов!
                    h, w = bayer.shape
                    h = h - (h % 2)
                    w = w - (w % 2)
                    bayer = bayer[:h, :w] # Отрезаем нечетный крайний пиксель
                    
                    # Нарезаем каналы (строго в правильном порядке RGGB)
                    r  = bayer[0::2, 0::2]
                    g1 = bayer[0::2, 1::2]
                    g2 = bayer[1::2, 0::2]
                    b  = bayer[1::2, 1::2]
                    
                    # Теперь NumPy склеит массивы без ошибок
                    lq = np.stack([r, g1, g2, b], axis=2)
                    lq_tensor = torch.from_numpy(lq.transpose(2, 0, 1)).float()
                    lq_tensor = T.functional.resize(lq_tensor, (256, 256))
                    input_tensor = lq_tensor.unsqueeze(0).to(device)

                # Инференс через модель
                output = model(input_tensor)
                if isinstance(output, dict):
                    output = output['out']

                # 🎯 Исправление: возвращаем к оригинальному размеру средствами PyTorch (без искажения каналов RGB)
                output_resized_tensor = T.functional.resize(output.squeeze(0), (orig_h, orig_w) if self.in_channels != 3 else (orig_h, orig_w))
                
                # Постпроцессинг тензора в numpy [H, W, C]
                output_np = output_resized_tensor.cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                final_img = (output_np * 255.0).astype(np.uint8)
                
                # Сохраняем файл с указанием эпохи 
                out_name = f"epoch_{epoch}_{base_name}.png"
                out_path = os.path.join(self.out_val_dir, out_name)
                Image.fromarray(final_img).save(out_path)
                
        # Возвращаем модель обратно в режим обучения перед выходом
        model.train()
