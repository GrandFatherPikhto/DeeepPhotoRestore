#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def smooth_exponential(data, alpha):
    if alpha >= 1.0:
        return data
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def load_csv_data(filepath, x_col, y_col, smoothing=1.0):
    df = pd.read_csv(filepath)
    x = df[x_col].values
    y = df[y_col].values
    if smoothing < 1.0:
        y = smooth_exponential(y, smoothing)
    mask = ~np.isnan(y)
    return x[mask], y[mask]

def load_log_data(filepath, pattern, smoothing=1.0):
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                values.append(float(match.group(1)))
    if not values:
        raise ValueError(f"Не найдено значений по паттерну {pattern} в {filepath}")
    x = np.arange(1, len(values)+1)
    y = np.array(values)
    if smoothing < 1.0:
        y = smooth_exponential(y, smoothing)
    return x, y

def plot_line(ax, x, y, title, xlabel, ylabel, y_lim=None, y_scale='linear', color=None):
    ax.plot(x, y, color=color or 'blue', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_lim:
        ax.set_ylim(y_lim)
    if y_scale == 'log':
        ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_comparison(ax, x, y, baseline_value, baseline_label, title, xlabel, ylabel, y_lim=None):
    ax.plot(x, y, color='blue', linewidth=1.5, label='NAFNet')
    ax.axhline(y=baseline_value, color='red', linestyle='--', linewidth=1.5, label=baseline_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

def plot_multi_line(ax, x, y_dict, title, xlabel, ylabel, y_scale='linear', legend=None):
    for i, (key, yvals) in enumerate(y_dict.items()):
        ax.plot(x, yvals, linewidth=1.5, label=legend[i] if legend else key)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_scale == 'log':
        ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5)
    if legend or len(y_dict) > 1:
        ax.legend()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', required=True, help='Путь к YAML конфигурации графиков')
    args = parser.parse_args()

    with open(args.opt, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    exp_dir = cfg['experiment_dir']
    out_dir = cfg.get('output_dir', os.path.join(exp_dir, 'figures'))
    os.makedirs(out_dir, exist_ok=True)

    data_files = cfg.get('data_files', {})

    for plot_cfg in cfg['plots']:
        plot_type = plot_cfg.get('type', 'line')
        data_cfg = plot_cfg['data']
        source = data_cfg.get('source', 'csv')

        if 'file_ref' in data_cfg:
            ref = data_cfg['file_ref']
            if ref not in data_files:
                print(f"Пропуск {plot_cfg.get('save')}: неизвестный file_ref '{ref}'")
                continue
            filepath = data_files[ref]
        elif 'file' in data_cfg:
            filepath = data_cfg['file']
        else:
            print(f"Пропуск {plot_cfg.get('save')}: не указан file или file_ref")
            continue

        filepath_abs = os.path.abspath(filepath)
        if not os.path.exists(filepath_abs):
            print(f"Пропуск {plot_cfg.get('save')}: файл {filepath_abs} не найден")
            continue

        try:
            if source == 'csv':
                if plot_type == 'multi_line':
                    # Загружаем DataFrame один раз
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