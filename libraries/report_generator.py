# -*- coding: utf-8 -*-
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Попробуем импортировать ssim из scikit-image для точного расчета
try:
    from skimage.metrics import structural_similarity as calculate_ssim
except ImportError:
    calculate_ssim = None


def load_image(path):
    try:
        img = Image.open(path)
        return img.convert('RGBA') if img.mode in ('RGBA', 'LA', 'P') else img.convert('RGB')
    except Exception as e:
        print(f"Ошибка загрузки {path}: {e}")
        sys.exit(1)


def resize_to_fit(img, target_size, bg_color):
    target_w, target_h = target_size
    scale = min(target_w / img.width, target_h / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    bg = Image.new(img_resized.mode, (target_w, target_h), bg_color)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    bg.paste(img_resized, (x, y))
    return bg


def add_border(img, border_config):
    if not border_config.get('enabled', False):
        return img
    thickness = border_config.get('thickness', 1)
    color = tuple(border_config.get('color', [0, 0, 0]))
    if img.mode == 'RGBA' and len(color) == 3:
        color = color + (255,)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(thickness):
        draw.rectangle([i, i, w-1-i, h-1-i], outline=color)
    return img


def ensure_rgba(img):
    return img if img.mode == 'RGBA' else img.convert('RGBA')


def compute_metrics(img_eval_path, img_gt_path, metrics_to_compute):
    """Вычисляет PSNR и SSIM между оцениваемым кадром и эталоном (HQ)."""
    metrics_text = []
    
    # Загружаем изображения для математики через PIL -> numpy
    try:
        img_eval = Image.open(img_eval_path).convert('RGB')
        img_gt = Image.open(img_gt_path).convert('RGB')
        
        # Приводим к одному размеру (к размеру GT), если они вдруг отличаются
        if img_eval.size != img_gt.size:
            img_eval = img_eval.resize(img_gt.size, Image.LANCZOS)
            
        arr_eval = np.array(img_eval)
        arr_gt = np.array(img_gt)
        
        # Расчет PSNR
        if 'psnr' in metrics_to_compute:
            mse = np.mean((arr_gt - arr_eval) ** 2)
            if mse == 0:
                psnr = 100.0
            else:
                psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
            metrics_text.append(f"PSNR: {psnr:.2f} dB")
            
        # Расчет SSIM
        if 'ssim' in metrics_to_compute:
            if calculate_ssim is not None:
                # Настройка channel_axis=2 для RGB картинок
                ssim_val = calculate_ssim(arr_gt, arr_eval, channel_axis=2)
            else:
                # Упрощенный fallback если scikit-image не установлен
                ssim_val = 0.0
                print("[WARNING] Для точного расчета SSIM установите: pip install scikit-image")
            metrics_text.append(f"SSIM: {ssim_val:.4f}")
            
    except Exception as e:
        print(f"Предупреждение: не удалось посчитать метрики для {img_eval_path}: {e}")
        
    return " | ".join(metrics_text)


def draw_text_on_image(img, text, config):
    """Накладывает плашку с текстом метрик в угол изображения."""
    if not text:
        return img
        
    draw = ImageDraw.Draw(img)
    
    # Пытаемся загрузить дефолтный шрифт покрупнее, если нет — берем стандартный мелкий
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default() # Для старых версий PIL
        
    # Считаем размеры текста для подложки
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    # Настройки отображения
    margin = 15
    padding = 6
    w, h = img.size
    
    # Позиция: по умолчанию 'top_left'
    position = getattr(config, 'metrics_position', 'top_left')
    if position == 'bottom_left':
        x = margin
        y = h - text_h - margin - padding * 2
    elif position == 'top_right':
        x = w - text_w - margin - padding * 2
        y = margin
    elif position == 'bottom_right':
        x = w - text_w - margin - padding * 2
        y = h - text_h - margin - padding * 2
    else: # top_left
        x = margin
        y = margin

    # Рисуем полупрозрачную подложку под текст (черная с альфой 160)
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ol_draw = ImageDraw.Draw(overlay)
    ol_draw.rectangle(
        [x, y, x + text_w + padding * 2, y + text_h + padding * 2], 
        fill=(0, 0, 0, 160)
    )
    
    # Объединяем слои
    img = Image.alpha_composite(ensure_rgba(img), overlay)
    
    # Пишем сам текст
    text_color = tuple(getattr(config, 'metrics_color', [255, 255, 255]))
    if len(text_color) == 3 and img.mode == 'RGBA':
        text_color = text_color + (255,)
        
    draw_final = ImageDraw.Draw(img)
    draw_final.text((x + padding, y + padding), text, fill=text_color, font=font)
    
    return img


def process_pair(img1_path, img2_path, output_path, config, hq_target_path=None):
    """Склеивает пару изображений и опционально выводит на них метрики."""
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    image_size = getattr(config, 'image_size', [512, 512])
    spacing = getattr(config, 'spacing', 10)
    bg_color = tuple(getattr(config, 'background_color', [255, 255, 255]))
    
    border_config = {
        'enabled': getattr(config, 'border_enabled', True),
        'color': getattr(config, 'border_color', [0, 0, 255]),
        'thickness': getattr(config, 'border_thickness', 1)
    }

    # Подготовка кадров (ресайз + рамка)
    left = add_border(resize_to_fit(img1, image_size, bg_color), border_config)
    right = add_border(resize_to_fit(img2, image_size, bg_color), border_config)

    # Определяем режим цветности
    need_alpha = (left.mode == 'RGBA' or right.mode == 'RGBA' or len(bg_color) == 4)
    if need_alpha:
        bg_final = bg_color + (255,) if len(bg_color) == 3 else bg_color
        mode = 'RGBA'
        left = ensure_rgba(left)
        right = ensure_rgba(right)
    else:
        mode = 'RGB'
        bg_final = bg_color

    total_width = image_size[0] * 2 + spacing
    total_height = image_size[1]

    # Собираем финальный холст
    result = Image.new(mode, (total_width, total_height), bg_final)
    if mode == 'RGBA':
        result.paste(left, (0, 0), left)
        result.paste(right, (image_size[0] + spacing, 0), right)
    else:
        result.paste(left, (0, 0))
        result.paste(right, (image_size[0] + spacing, 0))

    # --- РАБОТА С МЕТРИКАМИ ---
    metrics_enabled = getattr(config, 'metrics_enabled', True)
    metrics_list = getattr(config, 'metrics_list', ['psnr', 'ssim'])
    
    # Считаем метрики только если они включены и нам передан эталонный HQ-кадр
    if metrics_enabled and metrics_list and hq_target_path:
        # Для пары эпох (start->finish) оцениваем качество финишной эпохи (img2_path) относительно HQ
        # Для пары source (hq->lq) считать метрики бессмысленно, функция compute_metrics это поймет
        if "epochs" in str(output_path):
            text_score = compute_metrics(img2_path, hq_target_path, metrics_list)
            result = draw_text_on_image(result, text_score, config)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Склейка сохранена: {output_path}")
