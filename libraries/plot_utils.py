import re
import sys
import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

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

def substitute_name(obj, name):
    if isinstance(obj, dict):
        return {k: substitute_name(v, name) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_name(i, name) for i in obj]
    elif isinstance(obj, str):
        return obj.replace('{name}', name)
    else:
        return obj