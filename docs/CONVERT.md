# Конвертирование массива NEF

Для Ubuntu внутри WSL2 идеально подойдет связка из создания новой папки и утилиты mogrify с флагом -path.
В терминале WSL2 перейдите в папку с вашими картинками и выполните две команды:

mkdir -p output_jpg
mogrify -path ./output_jpg -format jpg *.NEF

(Если скачивали TIFF, то вместо *.NEF в конце укажите *.TIF)
## ⚠️ Важные нюансы для WSL2 / Ubuntu:

   1. Если команда ругается, что mogrify или magick не найден:
   Установите ImageMagick одной командой:
   
   sudo apt update && sudo apt install -y imagemagick
   
   2. Обязательно установите делегат для RAW (NEF):
   По умолчанию ImageMagick в Ubuntu не умеет читать файлы .NEF без стороннего декодера. Если при конвертации возникает ошибка no decode delegate for this image format, установите пакет ufraw-batch или dcraw:
   
   sudo apt install -y dcraw
   
   3. Если картинок очень много (ошибка Argument list too long):
   Если в папке тысячи файлов, команда *.NEF может выдать ошибку длины аргументов. В этом случае используйте безопасный обход через find:
   
   find . -maxdepth 1 -name "*.NEF" -exec mogrify -path ./output_jpg -format jpg {} +
   
   
Хотите добавить к этой конвертации автоматическое сжатие размера (например, уменьшить разрешение, чтобы сэкономить место на диске WSL)?

