import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt

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