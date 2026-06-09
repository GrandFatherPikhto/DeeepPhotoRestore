#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from libraries.plot_utils import (
    smooth_exponential, load_csv_data, load_log_data, substitute_name
)

from libraries.plot_functions import plot_line, plot_comparison, plot_multi_line

matplotlib.rcParams['agg.path.chunksize'] = 20000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', required=True, help='Путь к YAML конфигурации графиков')
    parser.add_argument('--name', type=str, default=None, help='Имя эксперимента (переопределяет {name})')
    args = parser.parse_args()

    with open(args.opt, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    exp_name = args.name if args.name is not None else cfg.get('name', '')
    if exp_name:
        cfg = substitute_name(cfg, exp_name)

    exp_dir = cfg['experiment_dir']
    out_dir = cfg.get('output_dir', os.path.join(exp_dir, 'figures'))
    os.makedirs(out_dir, exist_ok=True)

    data_files = cfg.get('data_files', {})

    for plot_cfg in cfg['plots']:
        plot_type = plot_cfg.get('type', 'line')
        data_cfg = plot_cfg['data']
        source = data_cfg.get('source', 'csv')

        # Определяем путь к файлу
        if source == 'val_csv':
            filepath_abs = os.path.join(exp_dir, 'val_metrics.csv')
            if not os.path.exists(filepath_abs):
                print(f"Пропуск {plot_cfg.get('save')}: файл {filepath_abs} не найден")
                continue
        elif 'file_ref' in data_cfg:
            ref = data_cfg['file_ref']
            if ref not in data_files:
                print(f"Пропуск {plot_cfg.get('save')}: неизвестный file_ref '{ref}'")
                continue
            filepath_abs = os.path.abspath(data_files[ref])
            if not os.path.exists(filepath_abs):
                print(f"Пропуск {plot_cfg.get('save')}: файл {filepath_abs} не найден")
                continue
        elif 'file' in data_cfg:
            filepath_abs = os.path.abspath(data_cfg['file'])
            if not os.path.exists(filepath_abs):
                print(f"Пропуск {plot_cfg.get('save')}: файл {filepath_abs} не найден")
                continue
        else:
            print(f"Пропуск {plot_cfg.get('save')}: не указан file или file_ref")
            continue

        try:
            if source in ('csv', 'val_csv'):
                if plot_type == 'multi_line':
                    df = pd.read_csv(filepath_abs)
                    x_col = data_cfg['x']
                    x = df[x_col].values
                    y_columns = data_cfg['y']
                    y_dict = {}
                    smoothing = plot_cfg.get('smoothing', 1.0)
                    for col in y_columns:
                        y_vals = df[col].values
                        if smoothing < 1.0:
                            y_vals = smooth_exponential(y_vals, smoothing)
                        y_dict[col] = y_vals
                    xlabel = data_cfg.get('x_label', x_col.capitalize())
                    ylabel = plot_cfg.get('y_label', 'Value')
                    fig, ax = plt.subplots(figsize=(8, 5))
                    plot_multi_line(ax, x, y_dict, plot_cfg['title'], xlabel, ylabel,
                                    y_scale=plot_cfg.get('y_scale', 'linear'),
                                    legend=plot_cfg.get('legend'))
                else:
                    x_col = data_cfg['x']
                    y_col = data_cfg['y']
                    smoothing = plot_cfg.get('smoothing', 1.0)
                    x, y = load_csv_data(filepath_abs, x_col, y_col, smoothing)
                    xlabel = data_cfg.get('x_label', x_col.capitalize())
                    ylabel = plot_cfg.get('y_label', y_col.capitalize())
                    fig, ax = plt.subplots(figsize=(8, 5))
                    if plot_type == 'line':
                        plot_line(ax, x, y, plot_cfg['title'], xlabel, ylabel,
                                  y_lim=plot_cfg.get('y_lim'),
                                  y_scale=plot_cfg.get('y_scale', 'linear'))
                    elif plot_type == 'comparison':
                        baseline_value = plot_cfg['baseline_value']
                        baseline_label = plot_cfg.get('baseline_label', 'Baseline')
                        plot_comparison(ax, x, y, baseline_value, baseline_label,
                                        plot_cfg['title'], xlabel, ylabel,
                                        y_lim=plot_cfg.get('y_lim'))
                    else:
                        raise ValueError(f"Unknown plot type for csv: {plot_type}")
            elif source == 'log':
                pattern = data_cfg['pattern']
                smoothing = plot_cfg.get('smoothing', 1.0)
                x, y = load_log_data(filepath_abs, pattern, smoothing)
                xlabel = data_cfg.get('x_label', 'Epoch')
                ylabel = plot_cfg.get('y_label', 'Value')
                fig, ax = plt.subplots(figsize=(8, 5))
                plot_line(ax, x, y, plot_cfg['title'], xlabel, ylabel,
                          y_lim=plot_cfg.get('y_lim'),
                          y_scale=plot_cfg.get('y_scale', 'linear'))
            else:
                raise ValueError(f"Unknown source: {source}")

            save_path = os.path.join(out_dir, plot_cfg['save'])
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Сохранён график: {save_path}")

        except Exception as e:
            print(f"Ошибка при построении {plot_cfg.get('save', 'неизвестного графика')}: {e}")
            continue

if __name__ == '__main__':
    main()