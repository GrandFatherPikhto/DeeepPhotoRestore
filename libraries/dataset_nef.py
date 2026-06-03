import os
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from PIL import Image
import torchvision.transforms.functional as F

class CustomNEFPairDataset(Dataset):
    def __init__(self, dataroot_lq, dataroot_gt):
        self.dataroot_lq = dataroot_lq
        self.dataroot_gt = dataroot_gt
        self.filenames = [os.path.splitext(f)[0].replace('_bayer', '') 
                          for f in os.listdir(dataroot_lq) if f.endswith('.tiff')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        
        # 1. Загружаем уже готовый 4-канальный 16-битный пакет от генератора
        lq_packed = tifffile.imread(os.path.join(self.dataroot_lq, f"{name}_bayer.tiff")).astype(np.float32) / 65535.0
        lq_tensor = torch.from_numpy(lq_packed.transpose(2, 0, 1)).float()

        # 2. Загружаем HQ таргет
        gt_path = os.path.join(self.dataroot_gt, f"{name}.png")
        gt_img = Image.open(gt_path).convert('RGB')
        gt_rgb = np.array(gt_img).astype(np.float32) / 255.0
        gt_tensor = torch.from_numpy(gt_rgb.transpose(2, 0, 1)).float()

        # Патчинг к тренировочному размеру
        lq_tensor = F.resize(lq_tensor, (256, 256), interpolation=F.InterpolationMode.BILINEAR)
        gt_tensor = F.resize(gt_tensor, (256, 256), interpolation=F.InterpolationMode.BILINEAR)

        return lq_tensor, gt_tensor
