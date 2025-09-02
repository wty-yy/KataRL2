#!/usr/bin/env python3
"""
Plots TensorBoard scalar data to compare different backends (e.g., gymnasium vs. envpool)
for the same environment.

This version is styled to be similar to DreamerV3 paper plots, featuring square-ish
subplots, larger fonts for publication, and a unified legend at the bottom. It uses a
robust layout method (tight_layout with a rect) to ensure proper spacing and prevent overlap.

Directory layout expected (example):
./logs/ppo/basic_discrete_cnn+mlp/
├── Assault-v5__envpool/
│   ├── seed_0_0/20250901-214317/tb/events.out.tfevents....
│   ├── seed_1_1/...
│   └── seed_2_2/...
├── Assault-v5__gymnasium/
│   ├── seed_0_0/...
│   └── ...
└── ...

Usage:
  python ./demos/common/plot_tb_graphs_compare_env_suits.py --logdir ./logs/ppo_100k_envpool_vs_gymnasium/basic_discrete_cnn+mlp
  python ./demos/common/plot_tb_graphs_compare_env_suits.py \
    --logdir ./logs/ppo/basic_discrete_cnn+mlp \
    --tag charts/episodic_return \
    --out ./logs/plots \
    --points 200 \
    --smooth 0.6

Notes:
- Requires `tensorboard` Python package. Install with: pip install tensorboard
- Uses only matplotlib.
"""

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Try to import TensorBoard's event accumulator; give a clear error if missing.
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    raise SystemExit(
        "Failed to import TensorBoard's EventAccumulator. "
        "Install it via `pip install tensorboard`.\n"
        f"Original error: {e}"
    )

# Regex to find seed directories like 'seed_0_0'
SEED_DIR_RE = re.compile(r"^seed_(\d+)_\d+$")
# Regex to parse 'EnvName__Backend' directories
ENV_BACKEND_RE = re.compile(r"(.+)__(.+)")

@dataclass
class ScalarSeries:
    steps: np.ndarray  # shape [N]
    values: np.ndarray  # shape [N]

def human_readable_formatter(x, pos):
    """Formats tick values into a human-readable string (e.g., 500K, 1M, 1.5M)."""
    if abs(x) >= 1_000_000:
        val_m = x / 1_000_000
        if val_m == int(val_m):
            return f'{int(val_m)}M'
        else:
            return f'{val_m:.1f}M'
    elif abs(x) >= 1_000:
        return f'{int(x / 1_000)}K'
    else:
        # For smaller numbers, show integer or one decimal place if not integer
        if x == int(x):
            return f'{int(x)}'
        else:
            return f'{x:.1f}'

def list_immediate_subdirs(path: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    except FileNotFoundError:
        return []

def find_tb_event_file(tb_dir: str) -> Optional[str]:
    if not os.path.isdir(tb_dir):
        return None
    candidates = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.startswith("events.out.tfevents")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def pick_latest_timestamp_dir(seed_dir: str) -> Optional[str]:
    ts_dirs = [d for d in list_immediate_subdirs(seed_dir) if re.match(r"^\d{8}-\d{6}$", d)]
    if not ts_dirs:
        return None
    ts_dirs.sort(reverse=True)
    return os.path.join(seed_dir, ts_dirs[0])

def load_scalar_series_from_event_file(event_path: str, tag: str) -> Optional[ScalarSeries]:
    try:
        ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return None
        events = ea.Scalars(tag)
        if not events:
            return None
        steps = np.array([e.step for e in events], dtype=np.float64)
        vals = np.array([e.value for e in events], dtype=np.float64)
        idx = np.argsort(steps)
        return ScalarSeries(steps=steps[idx], values=vals[idx])
    except Exception as e:
        print(f"[WARN] Failed reading {event_path}: {e}")
        return None

def ewma_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0 or len(y) < 2:
        return y
    s = np.empty_like(y)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = alpha * y[i] + (1 - alpha) * s[i - 1]
    return s

def aggregate_across_seeds(series_list: List[ScalarSeries], num_points: int, smooth_alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not series_list:
        return np.array([]), np.array([]), np.array([])

    min_max_step = min((s.steps.max() for s in series_list if len(s.steps) > 0), default=0)
    max_min_step = max((s.steps.min() for s in series_list if len(s.steps) > 0), default=0)

    if not np.isfinite(min_max_step) or not np.isfinite(max_min_step) or max_min_step >= min_max_step:
        ref_series = max(series_list, key=lambda s: len(s.steps), default=None)
        if ref_series is None or len(ref_series.steps) < 2:
            return np.array([]), np.array([]), np.array([])
        grid = np.linspace(ref_series.steps.min(), ref_series.steps.max(), num_points)
    else:
        grid = np.linspace(max_min_step, min_max_step, num_points)

    aligned_series = []
    for s in series_list:
        if len(s.steps) < 2:
            continue
        interp_values = np.interp(grid, s.steps, s.values)
        smoothed_values = ewma_smooth(interp_values, smooth_alpha)
        aligned_series.append(smoothed_values)

    if not aligned_series:
        return np.array([]), np.array([]), np.array([])

    stacked_series = np.vstack(aligned_series)
    mean = stacked_series.mean(axis=0)
    std = stacked_series.std(axis=0, ddof=0)
    return grid, mean, std

def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard data to compare backends (e.g., gymnasium vs. envpool).")
    parser.add_argument("--logdir", type=str, required=True, help="Directory containing 'EnvName__Backend' subfolders.")
    parser.add_argument("--tag", type=str, default="charts/episodic_return", help="Scalar tag to read from TensorBoard.")
    parser.add_argument("--out", type=str, default="./logs/plots", help="Output directory for figures.")
    parser.add_argument("--points", type=int, default=200, help="Number of points for interpolation grid.")
    parser.add_argument("--smooth", type=float, default=0.6, help="EWMA smoothing factor in [0,1). 0 disables.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    base_env_to_backend_series: Dict[str, Dict[str, List[ScalarSeries]]] = {}
    env_backend_dirs = [d for d in list_immediate_subdirs(args.logdir) if ENV_BACKEND_RE.match(d)]
    
    if not env_backend_dirs:
        print(f"[ERROR] No 'EnvName__Backend' style directories found in {args.logdir}.")
        return

    print("Scanning directories and loading TensorBoard data...")
    for env_backend_dir in tqdm(env_backend_dirs):
        match = ENV_BACKEND_RE.match(env_backend_dir)
        if not match: continue
        base_env, backend = match.groups()
        env_backend_path = os.path.join(args.logdir, env_backend_dir)
        seed_dirs = [d for d in list_immediate_subdirs(env_backend_path) if SEED_DIR_RE.match(d)]
        for seed_dir in seed_dirs:
            seed_path = os.path.join(env_backend_path, seed_dir)
            ts_dir = pick_latest_timestamp_dir(seed_path)
            if ts_dir is None: continue
            tb_dir = os.path.join(ts_dir, "tb")
            event_file = find_tb_event_file(tb_dir)
            if event_file is None: continue
            series = load_scalar_series_from_event_file(event_file, args.tag)
            if series is not None:
                base_env_to_backend_series.setdefault(base_env, {}).setdefault(backend, []).append(series)

    if not base_env_to_backend_series:
        print("[ERROR] No data loaded. Check --logdir and --tag.")
        return

    # --- 开始绘图 ---

    all_backends = sorted(list(set(
        backend for backend_map in base_env_to_backend_series.values() for backend in backend_map.keys()
    )))
    colors = plt.cm.tab10.colors
    color_map = {backend: colors[i % len(colors)] for i, backend in enumerate(all_backends)}

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 16,
        'lines.linewidth': 4.0,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
    })

    env_list = sorted(base_env_to_backend_series.keys())
    n_envs = len(env_list)
    if n_envs == 0:
        print("No environments with data to plot.")
        return
        
    ncols = min(n_envs, 4)
    nrows = math.ceil(n_envs / ncols)

    fig_width = 4 * ncols
    fig_height = 4 * nrows
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(fig_width, fig_height), 
        dpi=args.dpi, 
        squeeze=False,
        sharex=True, 
        sharey=False
    )
    axes = axes.flatten()

    print("\nGenerating overview figure...")
    for i, base_env in enumerate(env_list):
        ax = axes[i]
        backend_map = base_env_to_backend_series[base_env]
        
        for backend, series_list in sorted(backend_map.items()):
            grid, mean, std = aggregate_across_seeds(series_list, args.points, args.smooth)
            if grid.size > 0:
                ax.plot(grid, mean, label=backend, color=color_map[backend])
                ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=color_map[backend], linewidth=0)

        ax.set_title(base_env)
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(human_readable_formatter))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(human_readable_formatter))
        
        col = i % ncols
        # Only show Y-label on the first column
        if col == 0:
            ax.set_ylabel("Episodic Return")
        
        # Only show X-label on the last row
        if i >= n_envs - ncols:
            ax.set_xlabel("Env Step")
            # Explicitly turn on tick labels for the last row, as sharex might hide them
            ax.tick_params(axis='x', labelbottom=True)

    # Hide any unused subplots
    for i in range(n_envs, len(axes)):
        axes[i].set_visible(False)

    # Create a single, shared legend at the bottom
    legend_handles = [plt.Line2D([0], [0], color=color_map[backend], lw=3, label=backend) 
                      for backend in all_backends]
    
    fig.legend(handles=legend_handles, 
               loc='lower center',
               bbox_to_anchor=(0.5, 0.01), # Position the legend at the bottom center
               ncol=min(len(all_backends), 4), # Dynamic number of columns
               frameon=False)

    # Adjust layout to make space for the legend
    fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.5, h_pad=1.0)

    overview_path = os.path.join(args.out, "backend_comparison_overview.png")
    plt.savefig(overview_path)
    plt.close(fig)
    print(f"[OK] Saved overview figure: {overview_path}")


if __name__ == "__main__":
    main()