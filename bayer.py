import cv2
import numpy as np


def apply_bright_bayer(
    image_path,
    cell_size=8,
    pad=1,
    red_boost = 2.0,
    green_boost = 2.0,
    blue_boost=2.0,
    background_brightness=40,
    upscale_factor=1.0,
):
    # 1. Загружаем изображение
    img_src = cv2.imread(image_path)
    if img_src is None:
        print("Ошибка: Не удалось загрузить изображение.")
        return

    # 2. Делаем настраиваемый апскейл
    if upscale_factor != 1.0:
        img_src = cv2.resize(
            img_src,
            (0, 0),
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    # 3. Создаем фоновые подложки заданного серого цвета
    bg_color = [background_brightness] * 3
    bayer_r = np.full_like(img, bg_color)
    bayer_g = np.full_like(img, bg_color)
    bayer_b = np.full_like(img, bg_color)

    # 4. Шаг сетки (макропиксель 2х2 ячейки)
    step = cell_size * 2

    # 5. Проходим по изображению блоками
    for y in range(0, h - step + 1, step):
        for x in range(0, w - step + 1, step):

            # Координаты границ ячеек
            y0, y1, y2 = y, y + cell_size, y + cell_size * 2
            x0, x1, x2 = x, x + cell_size, x + cell_size * 2

            # Внутренние индексы с учетом отступа
            r_slice_y = slice(y0 + pad, y1 - pad)
            r_slice_x = slice(x0 + pad, x1 - pad)

            g1_slice_y = slice(y0 + pad, y1 - pad)
            g1_slice_x = slice(x1 + pad, x2 - pad)

            g2_slice_y = slice(y1 + pad, y2 - pad)
            g2_slice_x = slice(x0 + pad, x1 - pad)

            b_slice_y = slice(y1 + pad, y2 - pad)
            b_slice_x = slice(x1 + pad, x2 - pad)

            # --- TOP-LEFT: RED (с добавлением red_boost) ---
            bayer_r[r_slice_y, r_slice_x, :] = 0
            bayer_r[r_slice_y, r_slice_x, 0] = np.clip(
                img[r_slice_y, r_slice_x, 0] * red_boost, 0, 255
            )            

            # --- TOP-RIGHT: GREEN 1 ---
            bayer_g[g1_slice_y, g1_slice_x, :] = 0
            bayer_g[g1_slice_y, g1_slice_x, 1] = img[g1_slice_y, g1_slice_x, 1]

            # --- BOTTOM-LEFT: GREEN 2 ---
            bayer_g[g2_slice_y, g2_slice_x, :] = 0
            bayer_g[g2_slice_y, g2_slice_x, 1] = img[g2_slice_y, g2_slice_x, 1]

            # --- BOTTOM-RIGHT: BLUE ---
            bayer_b[b_slice_y, b_slice_x, :] = 0
            bayer_b[b_slice_y, b_slice_x, 2] = np.clip(
                img[b_slice_y, b_slice_x, 2] * blue_boost, 0, 255
            )

    # 6. Собираем общую мозаику
    final_bayer = np.full_like(img, bg_color)
    for channel in [bayer_r, bayer_g, bayer_b]:
        mask = np.any(channel != background_brightness, axis=-1)
        final_bayer[mask] = channel[mask]

    # 7. Сохраняем результаты
    cv2.imwrite("bayer_bright_red.png", cv2.cvtColor(bayer_r, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        "bayer_bright_green.png", cv2.cvtColor(bayer_g, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        "bayer_bright_blue.png", cv2.cvtColor(bayer_b, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        "bayer_bright_mosaic.png", cv2.cvtColor(final_bayer, cv2.COLOR_RGB2BGR)
    )

    print(
        f"✔️ Готово! Размер: {w}x{h} (Апскейл x{upscale_factor}), Ячейка: {cell_size}px"
    )


# ==========================================
# НАСТРОЙКИ СЕТКИ КОНФИГУРИРУЮТСЯ ЗДЕСЬ:
# ==========================================
INPUT_IMAGE = "/home/grand/Images/Tanahen/images.png"

UPSCALE = 4.0  # Во сколько раз увеличить картинку (например: 2.0, 4.0, 6.0)
SIZE = 8  # Размер цветной точки в пикселях
LINE = 1  # Толщина сетки
BLUE_BOOST = 2.5  # Высветление синего канала
RED_BOOST = 2.5 # 
GREEN_BOOST = 2.5 #
BG_YARKHOST = 50  # Яркость фона (серый цвет подложки)

apply_bright_bayer(
    INPUT_IMAGE,
    cell_size=SIZE,
    pad=LINE,
    red_boost=RED_BOOST,
    green_boost=GREEN_BOOST,
    blue_boost=BLUE_BOOST,
    background_brightness=BG_YARKHOST,
    upscale_factor=UPSCALE,
)
