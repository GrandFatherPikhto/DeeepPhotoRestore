import cv2
import numpy as np


def apply_large_bayer_filter(image_path, cell_size=6):
    # 1. Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка: Не удалось загрузить изображение.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    # 2. Инициализируем пустые (черные) холсты для каналов и мозаики
    bayer_r = np.zeros_like(img)
    bayer_g = np.zeros_like(img)
    bayer_b = np.zeros_like(img)

    # Вычисляем средний цвет в блоках, чтобы цвета не искажались при масштабировании
    # И заполняем крупную RGGB сетку
    for y in range(0, h - cell_size + 1, cell_size * 2):
        for x in range(0, w - cell_size + 1, cell_size * 2):

            # Границы для блоков 2х2 макропикселей
            y0, y1, y2 = y, y + cell_size, y + cell_size * 2
            x0, x1, x2 = x, x + cell_size, x + cell_size * 2

            # Берем оригинальные цвета из этих областей
            # Добавляем внутренний отступ (+1, -1), чтобы создать ЧЕРНУЮ рамку вокруг пикселя
            pad = 1 if cell_size > 2 else 0

            # --- TOP-LEFT: RED ---
            # Извлекаем только красный канал [..., 0], остальные каналы в этой зоне будут 0
            bayer_r[y0 + pad : y1 - pad, x0 + pad : x1 - pad, 0] = img[
                y0 + pad : y1 - pad, x0 + pad : x1 - pad, 0
            ]

            # --- TOP-RIGHT: GREEN (1) ---
            bayer_g[y0 + pad : y1 - pad, x1 + pad : x2 - pad, 1] = img[
                y0 + pad : y1 - pad, x1 + pad : x2 - pad, 1
            ]

            # --- BOTTOM-LEFT: GREEN (2) ---
            bayer_g[y1 + pad : y2 - pad, x0 + pad : x1 - pad, 1] = img[
                y1 + pad : y2 - pad, x0 + pad : x1 - pad, 1
            ]

            # --- BOTTOM-RIGHT: BLUE ---
            bayer_b[y1 + pad : y2 - pad, x1 + pad : x2 - pad, 2] = img[
                y1 + pad : y2 - pad, x1 + pad : x2 - pad, 2
            ]

    # 3. Собираем итоговую мозаику (сложение не пересекающихся по координатам массивов)
    final_bayer = bayer_r + bayer_g + bayer_b

    # 4. Конвертируем обратно в BGR для сохранения через OpenCV
    cv2.imwrite("large_bayer_red.png", cv2.cvtColor(bayer_r, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        "large_bayer_green.png", cv2.cvtColor(bayer_g, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite("large_bayer_blue.png", cv2.cvtColor(bayer_b, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        "large_bayer_mosaic.png", cv2.cvtColor(final_bayer, cv2.COLOR_RGB2BGR)
    )
    print(
        f"Готово! Все изображения сохранены (размер ячейки: {cell_size}px)."
    )


# Запуск скрипта (cell_size=6 сделает цветные точки шириной 6 пикселей с черной рамкой)
apply_large_bayer_filter("/home/grand/Images/Tanahen/images.png", cell_size=3)
