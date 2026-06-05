import os
import cv2
import torch
import numpy as np
import tifffile
from PIL import Image
import torchvision.transforms as T

from skimage.metrics import structural_similarity as ssim

from libraries.logger import get_logger


class VisualValidator:
    def __init__(self, opt):
        self.opt = opt
        self.logger = get_logger()
        log_cfg = self.opt.get('logger', {}).get('include_in_file_log', {})
        self.log_ssim_to_file = log_cfg.get('log_ssim', False)
        self.log_console_cfg = self.opt.get('logger', {}).get('include_in_console', {})
        self.in_channels = opt['network_g']['num_in_ch']
        exp_name = opt.get('name', 'default_experiment')
        
        dataset_root = opt['path']['dataset_root']
        self.hq_val_dir = os.path.join(dataset_root, "test", "hq_targets")
        self.lq_val_dir = os.path.join(dataset_root, "test", "lq_inputs")
        self.out_val_dir = os.path.join('experiments', exp_name, 'val_predictions')
        os.makedirs(self.out_val_dir, exist_ok=True)
        self.ext = '.jpg' if self.in_channels == 3 else '.tiff'

    def run_validation(self, model, epoch, device):
        """Прогоняет все тестовые файлы через модель и сохраняет результат"""
        if not os.path.exists(self.lq_val_dir):
            # print(f"⚠️ [Валидатор] Папка {self.lq_val_dir} не найдена. Пропускаем.")
            logger.warning(f"Папка {self.lq_val_dir} не найдена. Пропускаем.")
            return
            
        files = [f for f in os.listdir(self.lq_val_dir) if f.lower().endswith(self.ext)]
        if not files:
            # print(f"⚠️ [Валидатор] Нет файлов с расширением {self.ext} в {self.lq_val_dir}")
            logger.warning(f"Нет файлов с расширением {self.ext} в {self.lq_val_dir}")
            return
            
        model.eval()
        
        with torch.no_grad():
            for file_name in files:
                file_path = os.path.join(self.lq_val_dir, file_name)
                base_name = os.path.splitext(file_name)[0].replace('_bayer', '')
                
                if self.in_channels == 3:
                    # JPEG режим (3 канала)
                    input_img = Image.open(file_path).convert('RGB')
                    orig_w, orig_h = input_img.size
                    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
                    input_tensor = transform(input_img).unsqueeze(0).to(device)
                else:
                    # RAW режим: TIFF уже содержит 4 канала (RGGB)
                    lq = tifffile.imread(file_path).astype(np.float32) / 65535.0
                    orig_h, orig_w = lq.shape[:2]
                    # lq форма: (H, W, 4)
                    if lq.ndim != 3 or lq.shape[2] != 4:
                        raise ValueError(f"Ожидался 4-канальный TIFF, получен {lq.shape}")
                    lq_tensor = torch.from_numpy(lq.transpose(2, 0, 1)).float()
                    lq_tensor = T.functional.resize(lq_tensor, (256, 256))
                    input_tensor = lq_tensor.unsqueeze(0).to(device)

                output = model(input_tensor)
                if isinstance(output, dict):
                    output = output['out']

                output_resized_tensor = T.functional.resize(output.squeeze(0), (orig_h, orig_w))
                output_np = output_resized_tensor.cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                final_img = (output_np * 255.0).astype(np.uint8)

                # Загружаем эталонное изображение
                gt_path = os.path.join(self.hq_val_dir, f"{base_name}.png")
                if os.path.exists(gt_path):
                    gt_img = np.array(Image.open(gt_path).convert('RGB'))
                    # Приводим GT к размеру final_img (если нужно)
                    if gt_img.shape[:2] != final_img.shape[:2]:
                        gt_img = cv2.resize(gt_img, (final_img.shape[1], final_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    # Вычисляем SSIM (data_range=255, channel_axis=2)
                    ssim_val = ssim(final_img, gt_img, channel_axis=2, data_range=255)
                    # Логируем (можно использовать print, но лучше через logger)
                    logger = get_logger()
                    if self.log_ssim_to_file and os.path.exists(gt_path):
                        logger.info(f"[Валидатор] {base_name}: SSIM = {ssim_val:.4f}")
                    self.log_ssim_to_console = self.log_console_cfg.get('log_ssim', False)             
                
                out_name = f"epoch_{epoch}_{base_name}.png"
                out_path = os.path.join(self.out_val_dir, out_name)
                Image.fromarray(final_img).save(out_path)
                
        model.train()

    def calculate_ssim(self, restored_path, gt_path):
        restored = np.array(Image.open(restored_path).convert('RGB'))
        gt = np.array(Image.open(gt_path).convert('RGB'))
        # SSIM для каждого канала, затем среднее
        ssim_val = ssim(restored, gt, channel_axis=2, data_range=255)
        return ssim_val        