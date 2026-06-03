# PyTorch

## Установка и настройка [https://docs.pytorch.org/](https://docs.pytorch.org/)

Настройка PyTorch в Ubuntu 24.04 (Noble Numbat) внутри WSL2 имеет важную особенность: драйвер видеокарты ставится только в саму Windows, а WSL2 использует его автоматически через специальный мост. Внутри Ubuntu ничего из драйверов NVIDIA устанавливать не нужно. Проверяем доступность GPU командой:

### Шаг 1. Проверка связи с Windows-драйвером
Убедитесь, что Ubuntu в WSL2 видит вашу видеокарту NVIDIA. Выполните команду в терминале Ubuntu:

```text
nvidia-smi
Tue Jun  2 13:01:12 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.54                 Driver Version: 595.79         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   44C    P5             33W /  285W |    1536MiB /  12282MiB |     23%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

* Если таблица с видеокартой появилась — всё отлично, запишите версию CUDA из правого верхнего угла (обычно в WSL2 это 12.x) [1].
* Если команда не найдена / ошибка — вам нужно обновить официальный драйвер NVIDIA в самой Windows до актуальной версии и перезапустить WSL (wsl --shutdown в командной строке Windows) [1].

------------------------------
## Шаг 2. Установка системных зависимостей в Ubuntu
В Ubuntu 24.04 LTS для работы с окружениями Python и обработкой изображений (библиотека [https://github.com/opencv/opencv](OpenCV)) требуются системные пакеты.
В Ubuntu 24.04 (Noble Numbat) старый пакет libgl1-mesa-glx был удален. Вместо него для корректной работы библиотек обработки изображений (таких как OpenCV) нужно установить современные зависимости:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv libgl1 libglx-mesa0 libglib2.0-0
```
(Пакеты `libgl1` и `libglib2.0-0` критически важны — без них библиотеки восстановления картинок вроде OpenCV будут выдавать ошибку ImportError).
------------------------------
## Шаг 3. Создание изолированного окружения
В Ubuntu 24.04 запрещено устанавливать сторонние библиотеки глобально (защита externally-managed-environment). Создаем виртуальное окружение в вашей домашней директории:
```bash
cd ~/<Директория_Проекта>/Photo-Restoration-PyTorch
```

# Создаем окружение с именем torch_env
```bash
python3 -m venv .venv
```

# Активируем его
```bash
source ./.venv/bin/activate
```
Маркер (.venv) в начале строки подтверждает, что вы внутри.
------------------------------

## Шаг 4. Обновление pip и установка PyTorch
Установка и настройка актуального окружения [CUDA](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337)
Обновите пакетный менеджер и запустите установку. Выберите одну команду в зависимости от результата Шага 1:

CUDA 13.x — это самое свежее поколение архитектуры NVIDIA на сегодняшний день (актуально для 2026 года).

* Вариант А: Если nvidia-smi показал CUDA 12.x (рекомендуется для современных карт)
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

(Индекс cu124 стабильно работает на Ubuntu 24.04 и полностью совместим со всеми версиями драйверов CUDA 12.x в WSL2).
* Вариант Б: Если у вас нет дискретной видеокарты NVIDIA (работа только на CPU)

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
```

Объем загрузки составит около 2-3 ГБ, так как пакет содержит все необходимые математические ядра.

------------------------------
## Шаг 5. Проверка внутри Python
Можно сделать скрипт для стресс-теста `test_cuda.py`:

```python
#!./.venv/bin/python3
#test_cuda.py
import sysimport timeimport torch

def run_benchmark():
    print("=" * 50)
    print(" СИСТЕМНАЯ ИНФОРМАЦИЯ ".center(50, "="))
    print("=" * 50)
    print(f"Версия Python:       {sys.version.split()[0]}")
    print(f"Версия PyTorch:      {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"Доступность CUDA:    {'ВКЛЮЧЕНА' if cuda_available else 'ВЫКЛЮЧЕНА'}")

    if not cuda_available:
        print("\n[ОШИБКА]: GPU не доступен.")
        return

    print(f"Версия CUDA в Torch: {torch.version.cuda}")
    print(f"Имя устройства:      {torch.cuda.get_device_name(0)}")
    print(f"Память GPU (всего):  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    print("\n" + "=" * 50)
    print(" ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ".center(50, "="))
    print("=" * 50)

    device = torch.device("cuda")
    x = torch.randn(10000, 10000, device=device, dtype=torch.float32)
    y = torch.randn(10000, 10000, device=device, dtype=torch.float32)

    # Разогрев
    _ = torch.matmul(x, y)
    torch.cuda.synchronize()

    # Замер времени
    start_time = time.time()
    _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"Время выполнения на вашей RTX 4070 Ti: {(end_time - start_time) * 1000:.2f} мс")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark()
```

Можно из командной строки:
```bash
python
```
И копируем туда текст:
```python
import torch
print("PyTorch работает!")
print(f"Доступен ли GPU для вычислений: {torch.cuda.is_available()}")if torch.cuda.is_available():
    print(f"Ваша видеокарта в WSL2: {torch.cuda.get_device_name(0)}")
exit()
```
Если скрипт вывел `Доступен ли GPU для вычислений: True`, ваша среда полностью готова к запуску тяжелых нейросетей восстановления графики прямо из-под Ubuntu.
------------------------------

# PyTorch Hub

[https://pytorch.org/hub/](https://pytorch.org/hub/)

На официальном сайте [pytorch.org](pytorch.org) все готовые модели и инструкции к ним собраны в специальном каталоге, который называется [PyTorch Hub](https://pytorch.org/hub).Прямая ссылка на этот раздел: pytorch.org/hubКак устроен этот раздел и как им пользоваться:Поиск моделей: На главной странице Hub вы найдете категории: Компьютерное зрение (Vision), Текст (NLP), Аудио (Audio) и Генеративный ИИ (Generative).Готовый код: Кликнув на любую модель (например, ResNet, YOLO, BERT), вы попадете на страницу с подробным описанием и готовым блоком кода, который можно просто скопировать в свой проект.Прямая загрузка: Модели из этого раздела загружаются одной строчкой кода через функцию torch.hub.load без необходимости скачивать файлы вручную.Пример того, что вы найдете на сайте:Для загрузки модели классификации изображений (например, ResNeXt) сайт предложит вам следующий синтаксис:pythonimport torch

# Загрузка архитектуры и весов напрямую с сайта PyTorch
```python
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
model.eval()
```

Документация суб-библиотеки Torchvision: pytorch.org/vision/stable/models.html. 
Там находится самая актуальная таблица со списком всех доступных стандартных архитектур и их точностью (Accuracy).
Нужно выбрать модель или задачу.
Например, 
1. распознавание лиц, 
2. генерация картинок, 
3. перевод текста


Для восстановления изображений после искажений (сжатие, поворот, утрата пикселей) используются нейросети из областей Super-Resolution (сверхразрешение), Inpainting (дорисовка пикселей) и Image Restoration (общее восстановление).
На сайте pytorch.org и в официальном репозитории PyTorch Hub для этих задач доступны две мощные модели: Real-ESRGAN (для устранения артефактов сжатия и размытия) и HiFi-Face / другие генеративные модели (для дорисовки). Однако для комплексного искажения (с поворотом и потерей пикселей) лучшим решением в экосистеме PyTorch является библиотека Diffusers от Hugging Face или предобученные веса моделей NAER / Restormer / LaMa.
Ниже приведены готовые решения для каждого типа искажения.

------------------------------
## 1. Устранение артефактов сжатия и размытия (Real-ESRGAN)
Если картинка сильно сжата (например, в JPEG), на ней появляются «кубики» и шум. Модель Real-ESRGAN убирает эти артефакты и увеличивает четкость. В PyTorch Hub её можно загрузить через сторонние официальные репозитории.
```python
import torchimport cv2from PIL import Image
# Загрузка предобученной модели Real-ESRGAN через PyTorch Hub# (Автоматически скачивает архитектуру и веса)model = torch.hub.load('xinntao/Real-ESRGAN', 'RealESRGAN_x4plus', pretrained=True)
model.eval()
# Загрузка искаженного изображенияimg = cv2.imread('compressed_image.jpg')
# Восстановление (модель увеличит разрешение и уберет артефакты сжатия)with torch.no_grad():
    output, _ = model.enhance(img, outscale=4)

cv2.imwrite('restored_image.jpg', output)
```
------------------------------
## 2. Восстановление утраченных пикселей и дыр (Inpainting)
Если часть пикселей утеряна (черные пятна, битые области, удаленные объекты), применяется техника Inpainting. Самый современный и качественный способ в PyTorch — использование моделей латентной диффузии (Stable Diffusion Inpainting) через библиотеку diffusers.

# Перед запуском: ```bash pip install diffusers transformers torchimport torchfrom diffusers```
```python
import StableDiffusionInpaintingPipelinefrom PIL import Image
# Загрузка предобученной пайплайн-модели восстановленияpipe = StableDiffusionInpaintingPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", 
    torch_dtype=torch.float16
).to("cuda") # Рекомендуется использовать GPU
# Исходная поврежденная картинка и черно-белая маска утерянных пикселейinit_image = Image.open("damaged_image.png").convert("RGB")mask_image = Image.open("mask_of_lost_pixels.png").convert("RGB") # Белые зоны — то, что утеряно
# Восстановление утерянной частиprompt = "high quality restored photo, seamless, detailed"image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
image.save("restored_pixels.png")
```
Примечание: Если вам нужно строго математическое восстановление текстур без генерации нового контента ИИ, используйте модель LaMa (Large Mask Inpainting), ее веса также доступны на GitHub для PyTorch.
------------------------------
## 3. Исправление поворотов и геометрических искажений
Нейросети плохо справляются с «выпрямлением» повернутых картинок напрямую, так как это чисто геометрическая задача. Для этого в PyTorch используется Spatial Transformer Networks (STN) или классические методы, которые подготавливают картинку перед подачей в восстанавливающую нейросеть.
Если угол поворота неизвестен, его можно автоматически определить с помощью предобученной легкой сети RotNet, а затем выпрямить через встроенный модуль torchvision.transforms:
```python
import torchimport torchvision.transforms.functional as Ffrom PIL import Image
image = Image.open("rotated_and_damaged.jpg")
```
# 1. Геометрическое выравнивание (например, если модель определила поворот на 15 градусов)# В PyTorch это делается налету в тензорах без потери качества
```python
predicted_angle = -15.0 aligned_image = F.rotate(image, angle=predicted_angle, expand=True)
```
# 2. Далее aligned_image передается в модель Inpainting или Real-ESRGAN (см. шаги 1 и 2)

------------------------------
## Резюме: пайплайн комплексного восстановления
Чтобы восстановить изображение, где есть все три искажения одновременно, вам нужно построить конвейер (пайплайн):

   1. Выравнивание: Повернуть картинку обратно (torchvision.transforms.functional.rotate).
   2. Заполнение пустот: Создать маску утерянных пикселей и пропустить картинку через модель LaMa или Stable Diffusion Inpainting.
   3. Финальный апскейл: Пропустить результат через Real-ESRGAN, чтобы склеить границы, убрать шумы сжатия и вернуть резкость.

В каком формате у вас исходные данные (вы работаете с датасетом картинок или пишете single-скрипт)? Могу помочь объединить эти шаги в один рабочий скрипт.


## Установка библиотек работы с `CUDA`

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Базовые библиотеки (фреймворк) PyTorch с поддержкой видеокарт NVIDIA на базе технологии CUDA. [1, 2] 
Разбор параметров строки установки:

* torch: Главная библиотека для создания тензоров, построения и обучения любых нейросетей.
* torchvision: Дочерняя библиотека для работы с компьютерным зрением (содержит стандартные наборы данных, инструменты обработки изображений и популярные архитектуры вроде ResNet, YOLO и др.).
* --index-url https://download.pytorch.org/whl/cu126: Специальный репозиторий PyTorch, откуда скачиваются версии библиотек, скомпилированные под CUDA версии 12.6. [3, 4, 5, 6] 

## Зачем нужна именно эта команда?
Обычная команда pip install torch скачивает базовую версию (часто без поддержки GPU или с версией CUDA по умолчанию). Если у вас мощная видеокарта NVIDIA и установлены свежие драйверы с поддержкой CUDA 12.6, эта команда гарантирует, что ваши нейросети будут обучаться и восстанавливать картинки на видеокарте (GPU), а не на процессоре (CPU), что ускорит работу в десятки раз. [1, 5] 
## Как это связано с вашей задачей?
Чтобы запустить модели восстановления (например, Real-ESRGAN или Inpainting), о которых мы говорили ранее, вам обязательно нужно сначала развернуть сам фреймворк PyTorch с помощью этой команды.
После успешного выполнения этой строки в терминале вы сможете проверить, видит ли библиотека вашу видеокарту, выполнив в Python:
```python
import torch
print(torch.cuda.is_available()) # Должно вернуть True
```

[1] [https://northflank.com](https://northflank.com/blog/how-to-install-pytorch-for-production)
[2] [https://bitlaunch.io](https://bitlaunch.io/blog/how-to-install-pytorch-on-linux-and-windows/)
[3] [https://pypi.org](https://pypi.org/project/torch/)
[4] [https://www.youtube.com](https://www.youtube.com/watch?v=u7B8VPMcggA)
[5] [https://pytorch.org](https://pytorch.org/get-started/previous-versions/)
[6] [https://dev-discuss.pytorch.org](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337)

## Проверка ядер CUDA

```bash
python3 -c "import torch; print('CUDA доступна:', torch.cuda.is_available()); print('Версия CUDA в Torch:', torch.version.cuda); print('Устройство:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
Должно быть что-то вроде

```text
CUDA доступна: True
Версия CUDA в Torch: 12.6
Устройство: NVIDIA GeForce GTX 1060 6GB
```
