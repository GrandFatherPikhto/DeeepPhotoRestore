# ============================================================
# КОД ДЛЯ GOOGLE COLAB: ВИЗУАЛИЗАЦИЯ МАССИВА БАЙЕРА КАК НА СХЕМЕ
# ============================================================

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
from io import BytesIO
import matplotlib.patches as patches

# ============================================================
# 1. ЗАГРУЗКА И КРОП ИЗОБРАЖЕНИЯ
# ============================================================

print(">>> Нажмите 'Choose Files' и выберите 'demosaik bayer.png' <<<\n")

uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"Загружен файл: {filename}")

image_full = Image.open(BytesIO(uploaded[filename]))
image_full = np.array(image_full)

# Убираем альфа-канал, если он есть
if image_full.shape[2] == 4:
    image_full = image_full[:, :, :3]

# Убедимся, что тип uint8
if image_full.dtype != np.uint8:
    image_full = (image_full / image_full.max() * 255).astype(np.uint8)

# ------------------------------------------------------------
# НАСТРОЙКА КООРДИНАТ КВАДРАТА (Измените под ваше фото)
# ------------------------------------------------------------
# Откройте оригинал, прикиньте координаты нужной зоны в пикселях:
CROP_Y = 200     # Координата верхнего края квадрата (по вертикали)
CROP_X = 100     # Координата левого края квадрата (по горизонтали)
CROP_SIZE = 400  # Размер стороны квадрата (например, 100x100 пикселей)

# Вырезаем фрагмент из полноразмерной картинки
H_full, W_full = image_full.shape[:2]

# Защита от выхода за границы изображения
y_end = min(CROP_Y + CROP_SIZE, H_full)
x_end = min(CROP_X + CROP_SIZE, W_full)

image = image_full[CROP_Y:y_end, CROP_X:x_end]
# ------------------------------------------------------------

print(f"Полный размер фото: {image_full.shape[:2]}")
print(f"Размер выбранного фрагмента: {image.shape}")

# ============================================================
# 2. СОЗДАНИЕ МАСКИ БАЙЕРА И ПРИМЕНЕНИЕ
# ============================================================

def create_bayer_rggb(height, width):
    """
    Создание масок для паттерна RGGB:
    R  G
    G  B
    """
    y, x = np.mgrid[0:height, 0:width]

    r_mask = (y % 2 == 0) & (x % 2 == 0)   # чёт-чёт → R
    g_mask = ((y % 2 == 0) & (x % 2 == 1)) | ((y % 2 == 1) & (x % 2 == 0))  # чёт-нечёт, нечёт-чёт → G
    b_mask = (y % 2 == 1) & (x % 2 == 1)   # нечёт-нечёт → B

    return r_mask, g_mask, b_mask

def apply_bayer(image):
    """Наложение фильтра Байера на изображение"""
    h, w = image.shape[:2]
    r_mask, g_mask, b_mask = create_bayer_rggb(h, w)

    # Создаём выходное изображение
    bayer_img = np.zeros_like(image)

    # Применяем маски: оставляем только "свой" цвет, остальное 0
    bayer_img[:, :, 0] = image[:, :, 0] * r_mask  # R
    bayer_img[:, :, 1] = image[:, :, 1] * g_mask  # G
    bayer_img[:, :, 2] = image[:, :, 2] * b_mask  # B

    # Отдельные каналы (для визуализации)
    r_channel = np.zeros_like(image)
    g_channel = np.zeros_like(image)
    b_channel = np.zeros_like(image)

    r_channel[:, :, 0] = image[:, :, 0] * r_mask  # только R
    g_channel[:, :, 1] = image[:, :, 1] * g_mask  # только G
    b_channel[:, :, 2] = image[:, :, 2] * b_mask  # только B

    return bayer_img, r_channel, g_channel, b_channel, (r_mask, g_mask, b_mask)

# Применяем
bayer_img, r_ch, g_ch, b_ch, masks = apply_bayer(image)
r_mask, g_mask, b_mask = masks

# ============================================================
# 3. СОЗДАНИЕ ВИЗУАЛИЗАЦИИ КАК НА СХЕМЕ
# ============================================================

fig = plt.figure(figsize=(14, 16))

# --- ВЕРХ: МАСКА БАЙЕРА (паттерн) ---
ax1 = plt.subplot(3, 1, 1)

# Создаём визуализацию паттерна Байера
bayer_pattern = np.zeros((*r_mask.shape, 3), dtype=np.uint8)
bayer_pattern[:, :, 0] = r_mask * 255  # R
bayer_pattern[:, :, 1] = g_mask * 255  # G
bayer_pattern[:, :, 2] = b_mask * 255  # B

ax1.imshow(bayer_pattern)
ax1.set_title('Маска Байера (RGGB)', fontsize=14, fontweight='bold', pad=10)
ax1.axis('off')

# Стрелка вниз
ax1.annotate('', xy=(0.5, -0.05), xycoords='axes fraction',
            xytext=(0.5, -0.15), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# --- СЕРЕДИНА: ТРИ КАНАЛА ---
# Создаём подграфики для трёх каналов
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.1)

# Красный канал
ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(r_ch)
ax2.set_title('Красный канал (R)', fontsize=12, color='darkred', fontweight='bold')
ax2.axis('off')

# Зелёный канал
ax3 = fig.add_subplot(gs[1, 1])
ax3.imshow(g_ch)
ax3.set_title('Зелёный канал (G)', fontsize=12, color='darkgreen', fontweight='bold')
ax3.axis('off')

# Синий канал
ax4 = fig.add_subplot(gs[1, 2])
ax4.imshow(b_ch)
ax4.set_title('Синий канал (B)', fontsize=12, color='darkblue', fontweight='bold')
ax4.axis('off')

# Стрелка вниз (под каналами)
fig.text(0.5, 0.35, '↓', fontsize=30, ha='center', va='center', fontweight='bold')

# --- НИЗ: ИТОГОВОЕ ИЗОБРАЖЕНИЕ С БАЙЕРОМ ---
ax5 = fig.add_subplot(gs[2, :])
ax5.imshow(bayer_img)
ax5.set_title('Изображение с наложенным фильтром Байера', fontsize=14, fontweight='bold', pad=10)
ax5.axis('off')

plt.tight_layout()
plt.savefig('/content/bayer_scheme.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()

print("\n✓ Схема сохранена как 'bayer_scheme.png'")

# ============================================================
# 4. ДОПОЛНИТЕЛЬНО: ВАРИАНТ С УВЕЛИЧЕНИЕМ (как на вашей картинке)
# ============================================================

fig2, axes = plt.subplots(3, 3, figsize=(15, 15))

# Верхний ряд: маска Байера крупным планом (фрагмент)
h, w = image.shape[:2]
# Берём фрагмент 100x100 из центра или угла для наглядности
frag_size = min(300, h//4, w//4)
y0, x0 = h//2 - frag_size//2, w//2 - frag_size//2

# Маска (фрагмент)
axes[0, 1].imshow(bayer_pattern[y0:y0+frag_size, x0:x0+frag_size])
axes[0, 1].set_title('Маска Байера\n(фрагмент)', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# Пустые ячейки в верхнем ряду
axes[0, 0].axis('off')
axes[0, 2].axis('off')

# Стрелка
fig2.text(0.5, 0.63, '↓', fontsize=25, ha='center', va='center', fontweight='bold')

# Средний ряд: три канала (фрагменты)
axes[1, 0].imshow(r_ch[y0:y0+frag_size, x0:x0+frag_size])
axes[1, 0].set_title('R канал', fontsize=8, color='darkred', fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(g_ch[y0:y0+frag_size, x0:x0+frag_size])
axes[1, 1].set_title('G канал', fontsize=8, color='darkgreen', fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(b_ch[y0:y0+frag_size, x0:x0+frag_size])
axes[1, 2].set_title('B канал', fontsize=8, color='darkblue', fontweight='bold')
axes[1, 2].axis('off')

# Стрелка
fig2.text(0.5, 0.35, '↓', fontsize=25, ha='center', va='center', fontweight='bold')

# Нижний ряд: итоговое изображение (фрагмент)
axes[2, 1].imshow(bayer_img[y0:y0+frag_size, x0:x0+frag_size])
axes[2, 1].set_title('Байер-изображение\n(фрагмент)', fontsize=8, fontweight='bold')
axes[2, 1].axis('off')

axes[2, 0].axis('off')
axes[2, 2].axis('off')

plt.tight_layout()
plt.savefig('/content/bayer_scheme_fragment.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()

print("✓ Схема с фрагментами сохранена как 'bayer_scheme_fragment.png'")

# ============================================================
# 5. СОХРАНЕНИЕ И СКАЧИВАНИЕ
# ============================================================

# Сохраняем отдельные компоненты
cv2.imwrite('/content/bayer_mask.png', cv2.cvtColor(bayer_pattern, cv2.COLOR_RGB2BGR))
cv2.imwrite('/content/channel_R.png', cv2.cvtColor(r_ch, cv2.COLOR_RGB2BGR))
cv2.imwrite('/content/channel_G.png', cv2.cvtColor(g_ch, cv2.COLOR_RGB2BGR))
cv2.imwrite('/content/channel_B.png', cv2.cvtColor(b_ch, cv2.COLOR_RGB2BGR))
cv2.imwrite('/content/bayer_result.png', cv2.cvtColor(bayer_img, cv2.COLOR_RGB2BGR))

print("\n" + "="*50)
print("СОХРАНЁННЫЕ ФАЙЛЫ:")
print("="*50)
print("• bayer_scheme.png — полная схема")
print("• bayer_scheme_fragment.png — схема с фрагментами")
print("• bayer_mask.png — маска Байера")
print("• channel_R.png — красный канал")
print("• channel_G.png — зелёный канал")
print("• channel_B.png — синий канал")
print("• bayer_result.png — итоговое изображение")

# Скачивание
print("\n>>> Скачивание файлов...")
files.download('/content/bayer_scheme.png')
files.download('/content/bayer_scheme_fragment.png')

print("\n✓ Готово! Проверьте папку 'Загрузки'")