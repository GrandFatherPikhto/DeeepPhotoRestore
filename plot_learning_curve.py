#!./.venv/bin/python
import os
import argparse
import yaml
import matplotlib.pyplot as plt
from tbparse import SummaryReader

# --- ПАРСЕР АРГУМЕНТОВ ---
parser = argparse.ArgumentParser(description="Автоматическое построение кривых обучения на основе YAML")
parser.add_argument('-opt', type=str, required=True, help="Путь к конфигурационному файлу .yml")
args = parser.parse_args()

# --- ЗАГРУЗКА НАСТРОЕК ИЗ YML ---
with open(args.opt, 'r', encoding='utf-8') as f:
    opt = yaml.safe_load(f)

exp_name = opt.get('name', 'default_exp')
# Опираемся строго на конфигурацию путей!
exp_dir = os.path.join('experiments', exp_name)
tb_logs_dir = os.path.join(exp_dir, 'tb_logs')

print(f"🔍 Ищу логи TensorBoard в папке: {tb_logs_dir}")

if not os.path.exists(tb_logs_dir):
    print(f"❌ Ошибка: Папка с логами '{tb_logs_dir}' не найдена. Сначала запустите обучение.")
    exit(1)

# --- ЧТЕНИЕ ЛОГОВ С ПОМОЩЬЮ TBPARSE ---
try:
    reader = SummaryReader(tb_logs_dir)
    df = reader.scalars
except Exception as e:
    print(f"❌ Не удалось прочитать бинарные логи TensorBoard: {e}")
    exit(1)

if df.empty:
    print("⚠️ Логи TensorBoard пусты. Возможно, обучение только началось и данные ещё не сбросились на диск.")
    exit(0)

# Вытаскиваем уникальные теги метрик из лога (например, 'loss', 'psnr')
# Переводим в нижний регистр для надежного поиска
df['tag_lower'] = df['tag'].str.lower()

# Фильтруем данные по ключевым метрикам
loss_df = df[df['tag_lower'].str.contains('loss')]
psnr_df = df[df['tag_lower'].str.contains('psnr')]

if loss_df.empty or psnr_df.empty:
    print(f"⚠️ Доступные метрики в логе: {df['tag'].unique()}")
    print("Не удалось однозначно найти кривые Loss и PSNR. Проверьте имена тегов в вашем логгере.")
    exit(0)

# Сортируем по шагам, чтобы графики не ломались
loss_df = loss_df.sort_values('step')
psnr_df = psnr_df.sort_values('step')

# --- ПОСТРОЕНИЕ ГРАФИКОВ ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 1. График Функции Потерь (Loss)
ax1.plot(loss_df['step'], loss_df['value'], color='#E74C3C', label='Train Loss', alpha=0.5)
if len(loss_df) > 20:
    # Добавляем сглаживание скользящим средним
    smooth = loss_df['value'].rolling(window=min(20, len(loss_df)//5), min_periods=1).mean()
    ax1.plot(loss_df['step'], smooth, color='#C0392B', linewidth=2, label='Smoothed')
ax1.set_title(f'Кривая потерь ({loss_df["tag"].iloc[0]})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Глобальный шаг (Global Step)')
ax1.set_ylabel('Значение')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# 2. График Качества (PSNR)
ax2.plot(psnr_df['step'], psnr_df['value'], color='#2ECC71', label='Train PSNR', alpha=0.5)
if len(psnr_df) > 20:
    smooth = psnr_df['value'].rolling(window=min(20, len(psnr_df)//5), min_periods=1).mean()
    ax2.plot(psnr_df['step'], smooth, color='#27AE60', linewidth=2, label='Smoothed')
ax2.set_title(f'Метрика качества ({psnr_df["tag"].iloc[0]})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Глобальный шаг (Global Step)')
ax2.set_ylabel('dB')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.suptitle(f"Мониторинг эксперимента: {exp_name}", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

# Сохраняем готовую картинку прямо в папку эксперимента рядом с весами!
output_plot_path = os.path.join(exp_dir, 'learning_curves.png')
plt.savefig(output_plot_path, dpi=150)
print(f"🎉 График успешно построен и сохранен в: {output_plot_path}")
