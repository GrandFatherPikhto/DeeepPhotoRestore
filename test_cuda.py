#!./.venv/bin/python
#
import sys
import time
import torch


def run_benchmark():
    print("=" * 50)
    print(" СИСТЕМНАЯ ИНФОРМАЦИЯ ".center(50, "="))
    print("=" * 50)
    print(f"Версия Python:       {sys.version.split()[0]}")
    print(f"Версия PyTorch:      {torch.__version__}")

    # Проверка доступности CUDA
    cuda_available = torch.cuda.is_available()
    print(f"Доступность CUDA:    {'ВКЛЮЧЕНА (True)' if cuda_available else 'ВЫКЛЮЧЕНА (False)'}")

    if not cuda_available:
        print("\n[ВНИМАНИЕ]: GPU не доступен. Вычисления будут идти на CPU.")
        print("=" * 50)
        return

    # Информация о видеокарте
    print(f"Версия CUDA в Torch: {torch.version.cuda}")
    print(f"Имя устройства:      {torch.cuda.get_device_name(0)}")
    print(f"Количество GPU:      {torch.cuda.device_count()}")

    # Память видеокарты
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Память GPU (всего):  {total_mem:.2f} GB")

    print("\n" + "=" * 50)
    print(" ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ (БЕНЧМАРК) ".center(50, "="))
    print("=" * 50)
    print("Запуск стресс-теста: перемножение матриц 10000x10000...")

    try:
        # Выделяем память под огромные матрицы на GPU
        device = torch.device("cuda")
        x = torch.randn(10000, 10000, device=device, dtype=torch.float32)
        y = torch.randn(10000, 10000, device=device, dtype=torch.float32)

        # Разогрев GPU (первый запуск всегда медленнее из-за инициализации)
        _ = torch.matmul(x, y)
        torch.cuda.synchronize()

        # Замер времени
        start_time = time.time()

        # Выполняем операцию
        result = torch.matmul(x, y)

        # Ждем завершения операции на GPU (асинхронность CUDA)
        torch.cuda.synchronize()
        end_time = time.time()

        print(f"Успешно! Матрицы перемножены.")
        print(f"Время выполнения на GPU: {(end_time - start_time) * 1000:.2f} мс")

        # Очистка памяти
        del x, y, result
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Ошибка во время теста производительности: {e}")

    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()
