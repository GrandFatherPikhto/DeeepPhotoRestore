#!./.venv/bin/python
#
"""
Скрипт для разбиения содержимого Python-файлов на блоки заданного размера
с удалением пустых строк и комментариев.

Читает список файлов и директорий из текстового файла (по одному пути на строку).
Для директорий рекурсивно находит все файлы с расширением .py.
Из каждого .py файла удаляются пустые строки и строки-комментарии (начинающиеся с #).
Оставшиеся строки разбиваются на блоки по N строк (по умолчанию 50).
Для каждого блока в выходной файл записывается заголовок (относительный путь к файлу и номер части)
и сам блок.
"""

import os
import sys
import argparse
from typing import List, Iterator, Tuple


def read_paths_from_file(list_file: str) -> List[str]:
    """
    Читает список путей из текстового файла.
    Пропускает пустые строки и строки-комментарии (начинающиеся с #).
    """
    paths = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            paths.append(line)
    return paths


def collect_py_files(paths: List[str]) -> Iterator[str]:
    """
    Собирает все .py файлы из переданных путей.
    Если путь – директория, обходит её рекурсивно.
    """
    for path in paths:
        if not os.path.exists(path):
            print(f"Предупреждение: путь не существует - {path}", file=sys.stderr)
            continue

        if os.path.isfile(path):
            if path.endswith('.py'):
                yield os.path.abspath(path)
            else:
                print(f"Предупреждение: пропущен файл (не .py) - {path}", file=sys.stderr)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        yield os.path.abspath(os.path.join(root, file))
        else:
            print(f"Предупреждение: неизвестный тип пути - {path}", file=sys.stderr)


def filter_lines(lines: List[str]) -> List[str]:
    """
    Удаляет пустые строки и строки-комментарии (первый непробельный символ #).
    """
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:          # пустая строка
            continue
        if stripped.startswith('#'):  # строка-комментарий
            continue
        filtered.append(line.rstrip('\n'))  # сохраняем без \n, добавим позже
    return filtered


def chunks_from_file(file_path: str, block_size: int) -> Iterator[Tuple[int, List[str]]]:
    """
    Читает файл, удаляет пустые строки и комментарии, возвращает блоки строк.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    significant_lines = filter_lines(original_lines)
    if not significant_lines:
        return  # нет значимых строк — пропускаем файл

    num_chunks = (len(significant_lines) + block_size - 1) // block_size
    for part_num in range(1, num_chunks + 1):
        start = (part_num - 1) * block_size
        end = start + block_size
        chunk = significant_lines[start:end]
        yield part_num, chunk


def main():
    parser = argparse.ArgumentParser(
        description="Разбивает содержимое .py файлов на блоки строк (без пустых строк и комментариев) "
                    "и записывает в выходной файл. Список путей читается из текстового файла."
    )
    parser.add_argument(
        'list_file',
        help="Текстовый файл со списком файлов и/или директорий (по одному пути на строку)"
    )
    parser.add_argument(
        '--block-size', '-b',
        type=int,
        default=50,
        help="Количество значимых строк в одном блоке (по умолчанию 50)"
    )
    parser.add_argument(
        '--output', '-o',
        default='output.txt',
        help="Имя выходного файла (по умолчанию output.txt)"
    )
    args = parser.parse_args()

    # Читаем список путей из указанного файла
    if not os.path.exists(args.list_file):
        print(f"Ошибка: файл со списком не найден - {args.list_file}", file=sys.stderr)
        sys.exit(1)

    paths = read_paths_from_file(args.list_file)
    if not paths:
        print("Файл со списком не содержит ни одного пути (или только комментарии/пустые строки).",
              file=sys.stderr)
        sys.exit(1)

    # Собираем все .py файлы
    py_files = list(collect_py_files(paths))
    if not py_files:
        print("Не найдено ни одного .py файла для обработки.", file=sys.stderr)
        sys.exit(1)

    current_dir = os.getcwd()
    chunk_num = 0
    with open(args.output, 'w', encoding='utf-8') as out_f:
        for abs_path in py_files:
            # Получаем относительный путь от текущей рабочей директории
            try:
                rel_path = os.path.relpath(abs_path, start=current_dir)
            except ValueError:
                # на разных дисках в Windows – оставляем абсолютный путь
                rel_path = abs_path

            # Разбиваем файл на блоки
            for part_num, chunk in chunks_from_file(abs_path, args.block_size):
                # Заголовок блока
                out_f.write(f"## === {rel_path} (part {part_num}) ===\n")
                # Строки блока (каждая записывается с переводом строки)
                for line in chunk:
                    out_f.write(line + '\n')
                # Разделитель между блоками
                out_f.write('\n')
                
                chunk_num += 1

    print(f"Готово. Результат записан в {args.output} всего чанков {chunk_num}")


if __name__ == '__main__':
    main()
