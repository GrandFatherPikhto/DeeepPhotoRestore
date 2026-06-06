# Установка и настройка окружения для работы скриптов

## 1. Создание виртуального окружения

```bash
python3 -m venv .venv
```

## 2. Активация виртуального окружения

- **Linux / macOS / WSL2**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows (CMD)**:
  ```bash
  .venv\Scripts\activate.bat
  ```
- **Windows (PowerShell)**:
  ```bash
  .venv\Scripts\Activate.ps1
  ```

*Примечание:* В командной строке должен появиться индикатор `(.venv)`. Если его нет – виртуальное окружение не активировано.

## 3. Обновление pip и установка PyTorch с CUDA

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

> **Важно:** Установка без CUDA (только CPU) возможна, но обучение будет крайне медленным. Рекомендуется использовать GPU NVIDIA с драйверами CUDA 12.x.

## 4. Установка остальных зависимостей

Убедитесь, что в корне проекта есть файл `requirements.txt`. Установите все необходимые библиотеки:

```bash
pip install -r requirements.txt
```

## 5. Проверка работоспособности PyTorch и CUDA

### Способ А. Интерактивная проверка (REPL)

```bash
python3
```

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
exit()
```

### Способ Б. Скрипт комплексного тестирования и бенчмаркинга

Создайте файл `test_cuda.py` со следующим содержимым:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import torch

def run_benchmark():
    print("=" * 50)
    print(" SYSTEM INFORMATION ".center(50, "="))
    print("=" * 50)
    print(f"Python version:   {sys.version.split()[0]}")
    print(f"PyTorch version:  {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available:   {'YES' if cuda_available else 'NO'}")

    if not cuda_available:
        print("\n[ERROR] GPU not accessible. Training will be slow.")
        return

    device_props = torch.cuda.get_device_properties(0)
    total_memory_gb = device_props.total_memory / (1024 ** 3)

    print(f"CUDA version (Torch): {torch.version.cuda}")
    print(f"GPU name:            {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:          {total_memory_gb:.2f} GB")

    print("\n" + "=" * 50)
    print(" PERFORMANCE TEST ".center(50, "="))
    print("=" * 50)

    device = torch.device("cuda")
    x = torch.randn(10000, 10000, device=device, dtype=torch.float32)
    y = torch.randn(10000, 10000, device=device, dtype=torch.float32)

    # Warm-up
    _ = torch.matmul(x, y)
    torch.cuda.synchronize()

    start = time.time()
    _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    end = time.time()

    ms = (end - start) * 1000
    print(f"Matrix multiplication (10000x10000): {ms:.2f} ms")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark()
```

Запустите скрипт:

```bash
python test_cuda.py
```

Если CUDA доступна, вы увидите данные о GPU и время выполнения.

## 6. Интеграция архитектуры NAFNet (как Git‑субмодуль)

Проект использует модифицированную версию библиотеки `basicsr` из официального репозитория [NAFNet](https://github.com/megvii-research/NAFNet). Не устанавливайте `basicsr` через `pip` – она конфликтует.

### 6.1 Добавление субмодуля

```bash
git submodule add -f https://github.com/megvii-research/NAFNet.git modules/NAFNet
cd modules/NAFNet
git submodule update --init --recursive
```

**Зачем `-f`?** Принудительно добавляет субмодуль, даже если папка уже существует (полезно при повторной настройке).

### 6.2 Настройка `.gitmodules` (опционально)

Чтобы избежать случайных коммитов изменений в субмодуле, добавьте в `.gitmodules`:

```ini
[submodule "modules/NAFNet"]
    path = modules/NAFNet
    url = https://github.com/megvii-research/NAFNet.git
    ignore = all
```

### 6.3 Удаление глобального `basicsr` (если установлен)

```bash
pip uninstall -y basicsr
```

### 6.4 Локальная сборка NAFNet

Перейдите в папку субмодуля и выполните установку:

```bash
cd modules/NAFNet
python setup.py develop --no_cuda_ext
```

> **Обязательно** используйте флаг `--no_cuda_ext`. Он предотвращает компиляцию C++/CUDA расширений, которые часто ломаются в WSL2 или без полного набора инструментов NVidia.

После этого в вашем окружении появится символическая ссылка на папку `basicsr` внутри `modules/NAFNet`. Все импорты `from basicsr...` будут работать корректно.

Вернитесь в корень проекта:

```bash
cd ../..
```

## 7. Проверка работоспособности всего пайплайна (опционально)

Убедитесь, что все модули импортируются без ошибок:

```bash
python -c "from libraries.config import get_pipeline_config; print('OK')"
```

Если нет ошибок – среда настроена.

## 8. Дополнительные замечания

- **Активация окружения** при каждом новом сеансе работы обязательна. Для выхода из виртуального окружения выполните `deactivate`.
- **Требования к GPU** – рекомендуется NVIDIA RTX 3060 или лучше с 12 ГБ VRAM. На CPU обучение крайне медленное.
- **Если возникают ошибки `ModuleNotFoundError`** – проверьте, что вы находитесь в корне проекта и активировали виртуальное окружение.
- **Документация по использованию** – см. [README.md](README.md) и файлы в папке `docs/`.

---

**После установки** вы можете приступить к генерации датасета и обучению модели. Следующие шаги описаны в [PIPELINE.md](docs/PIPELINE.md) и [TRAINING.md](docs/TRAINING.md).
