#!/usr/bin/env python3
"""
Plot TensorBoard `charts/episodic_return` aggregated by seed, grouped by environment
(one figure per environment) and with one curve per algorithm. The mean is plotted
as a solid line and the standard deviation is shown as a translucent band.

This version is styled to be similar to DreamerV3 paper plots, featuring square-ish
subplots, larger fonts for publication, and a unified legend at the bottom. It uses a
robust layout method (tight_layout with a rect) to ensure proper spacing and prevent overlap.

A summary plot can be added to compare normalized performance across all environments.

Directory layout expected (example):
./logs/
├── sac/
│   ├── basic_continuous_mlp/
│   │   └── Hopper-v4__gymnasium/
│   │   │   ├── seed_0_0/20250811-234258/tb/events.out.tfevents....
│   │   │   ├── seed_1_1/...
│   │   │   └── seed_2_2/...
│   │   └── Hopper-v4__envpool/
│   │        └── ...
│   └── simba_continuous_mlp/
│        └── ...
└── ppo/
     └── ...

Usage: (details see assets/figures/plot_commands.md)
python ./demos/common/plot_tb_graphs.py --avail-algos ppo sac
python ./demos/common/plot_tb_graphs.py --avail-algos ppo sac --ignore-env-suit-name
python ./demos/common/plot_tb_graphs.py --avail-algos ppo --ignore-env-suit-name --avail-suits envpool --avail-envs Breakout-v5

python ./demos/common/plot_tb_graphs.py --logdir ./logs --tag charts/episodic_return --out ./logs/plots --points 100 --smooth 0.5 --avail-algos sac

Notes:
- Requires `tensorboard` Python package (for EventAccumulator). Install with: pip install tensorboard
- Uses only matplotlib (no seaborn).
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
import matplotlib.ticker as mticker # 用于格式化坐标轴

# Try to import TensorBoard's event accumulator; give a clear error if missing.
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    raise SystemExit(
        "Failed to import TensorBoard's EventAccumulator. "
        "Install it via `pip install tensorboard`.\n"
        f"Original error: {e}"
    )

SEED_DIR_RE = re.compile(r"^(seed_\d+_\d+)$")

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
        val_m = x / 1_000
        if val_m == int(val_m):
            return f'{int(x / 1_000)}K'
        else:
            return f'{val_m:.1f}K'
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
        vals  = np.array([e.value for e in events], dtype=np.float64)
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
        s[i] = alpha * y[i] + (1 - alpha) * s[i-1]
    return s

def aggregate_series(series_list: List[ScalarSeries], num_points: int, smooth_alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregates a list of ScalarSeries into a mean and std dev curve on a common grid."""
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

    aligned = []
    for s in series_list:
        if len(s.steps) < 2:
            continue
        vals = np.interp(grid, s.steps, s.values)
        if smooth_alpha > 0:
            vals = ewma_smooth(vals, smooth_alpha)
        aligned.append(vals)

    if not aligned:
        return np.array([]), np.array([]), np.array([])

    A = np.vstack(aligned)
    mean = A.mean(axis=0)
    std  = A.std(axis=0, ddof=0)
    return grid, mean, std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./logs", help="Root logs directory.")
    parser.add_argument("--tag", type=str, default="charts/episodic_return", help="Scalar tag to read from TB.")
    parser.add_argument("--out", type=str, default="./logs/plots", help="Output directory for figures.")
    parser.add_argument("--points", type=int, default=100, help="Number of points on the common grid.")
    parser.add_argument("--smooth", type=float, default=0.5, help="EWMA smoothing factor in [0,1). 0 disables.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    parser.add_argument("--avail-algos", nargs='+', default=['sac'], help="List of available algorithms.")
    parser.add_argument("--avail-family-algos", nargs='+', default=None, help="List of available algorithm families (subfolders).")
    parser.add_argument("--avail-envs", nargs='+', default=None, help="List of available environments.")
    parser.add_argument("--avail-suits", nargs='+', default=None, help="List of available environment suites.")
    parser.add_argument("--ignore-env-suit-name", action='store_true', help="Ignore environment suite name (e.g. gymnaisum, dmc, ...).")
    parser.add_argument("--plot-summary", action='store_true', help="Also plot a summary figure with all envs.")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    algos = list_immediate_subdirs(args.logdir)
    algos = [a for a in algos if a in args.avail_algos]
    if not algos:
        print(f"[ERROR] No algorithm folders matching --avail-algos under {args.logdir}.")
        return

    # Data structure for overview plot: env -> algo -> [series_per_seed]
    env_to_algo_series: Dict[str, Dict[str, List[ScalarSeries]]] = {}
    # Data structure for summary plot: seed -> algo -> [series_per_env]
    seed_to_algo_series: Dict[str, Dict[str, List[ScalarSeries]]] = {}

    env_algo_fam_paths = []
    for algo in algos:
        algo_root = os.path.join(args.logdir, algo)
        families = list_immediate_subdirs(algo_root)
        for fam in families:
            if args.avail_family_algos and fam not in args.avail_family_algos:
                continue
            fam_root = os.path.join(algo_root, fam)
            envs = list_immediate_subdirs(fam_root)
            for env in envs:
                env_id, env_suit = env.split('__')
                if args.avail_envs and env_id not in args.avail_envs:
                    continue
                if args.avail_suits and env_suit not in args.avail_suits:
                    continue
                env_root = os.path.join(fam_root, env)
                seed_dirs = [d for d in list_immediate_subdirs(env_root) if SEED_DIR_RE.match(d)]
                for seed_dir in seed_dirs:
                    seed_path = os.path.join(env_root, seed_dir)
                    env_algo_fam_paths.append([env, algo, fam, seed_dir, seed_path])
    
    print("Scanning directories and loading data...")
    for env, algo, fam, seed_dir, seed_path in tqdm(env_algo_fam_paths):
        ts_dir = pick_latest_timestamp_dir(seed_path)
        if ts_dir is None: continue
        tb_dir = os.path.join(ts_dir, "tb")
        event_file = find_tb_event_file(tb_dir)
        if event_file is None: continue
        series = load_scalar_series_from_event_file(event_file, args.tag)
        if series is None: continue
        
        algo_label = f"{algo}_{fam}"
        env_to_algo_series.setdefault(env, {}).setdefault(algo_label, []).append(series)
        if args.plot_summary:
            seed_to_algo_series.setdefault(seed_dir, {}).setdefault(algo_label, []).append(series)

    if not env_to_algo_series:
        print("[ERROR] No data found. Check your paths and arguments.")
        return

    all_algos_labels = sorted(list(set(
        algo for algo_map in env_to_algo_series.values() for algo in algo_map.keys()
    )))

    # --- Plotting settings ---
    colors = plt.cm.tab10.colors
    color_map = {algo: colors[i % len(colors)] for i, algo in enumerate(all_algos_labels)}
    plt.rcParams.update({
        'font.size': 14, 'axes.titlesize': 20 if args.ignore_env_suit_name else 16,
        'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 16, 'lines.linewidth': 4.0, 'grid.linewidth': 0.5,
        'grid.linestyle': '--', 'grid.alpha': 0.6, 'figure.titlesize': 22,
    })

    # --- Generate and save the main overview figure (per-environment) ---
    env_list = sorted(env_to_algo_series.keys())
    n_envs = len(env_list)
    if n_envs > 0:
        ncols = min(n_envs, 4)
        nrows = math.ceil(n_envs / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=args.dpi, squeeze=False, sharex=True, sharey=False)
        axes = axes.flatten()

        print("\nGenerating overview figure (per-environment performance)...")
        for i, env in enumerate(env_list):
            ax = axes[i]
            algo_map = env_to_algo_series[env]
            for algo, series_list in sorted(algo_map.items()):
                grid, mean, std = aggregate_series(series_list, args.points, args.smooth)
                if grid.size > 0:
                    ax.plot(grid, mean, label=algo, color=color_map[algo])
                    ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=color_map[algo], linewidth=0)
            
            env_title = env.split('__')[0] if args.ignore_env_suit_name else env
            ax.set_title(env_title)
            ax.grid(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(human_readable_formatter))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(human_readable_formatter))
            if i % ncols == 0: ax.set_ylabel("Episodic Return")
            if (i // ncols) == nrows - 1:
                ax.set_xlabel("Env Step")
                ax.tick_params(axis='x', rotation=30)

        for i in range(n_envs, len(axes)): axes[i].set_visible(False)
        all_algos_abbr_labels = [re.sub(r'(_continuous|_discrete)', '', algo) for algo in all_algos_labels]
        legend_handles = [plt.Line2D([0], [0], color=color_map[algo], lw=3, label=abbr_algo) for algo, abbr_algo in zip(all_algos_labels, all_algos_abbr_labels)]
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=min(len(all_algos_labels), 4), frameon=False)
        if args.title:
            fig.suptitle(args.title)
        fig.tight_layout(rect=[0.0, 0.05, 1, 1], w_pad=0.8, h_pad=1.5)

        overview_path = os.path.join(args.out, "all_envs_overview.png")
        plt.savefig(overview_path)
        plt.close(fig)
        print(f"[OK] Saved overview figure: {overview_path}")
    else:
        print("No environments with data to plot for the main overview.")


    # --- Generate and save the summary figure (new logic) ---
    if args.plot_summary and seed_to_algo_series:
        print("\nGenerating summary figure (new logic)...")
        # Step 1: For each algo, create a list of per-seed summary curves
        algo_to_seed_summary_series: Dict[str, List[ScalarSeries]] = {algo: [] for algo in all_algos_labels}
        print("Step 1/2: Averaging across environments for each seed...")
        for seed_id, algo_map in tqdm(seed_to_algo_series.items()):
            for algo, env_curves in algo_map.items():
                # Aggregate all environment curves for this specific seed and algo
                grid, mean, _ = aggregate_series(env_curves, args.points, args.smooth)
                if grid.size > 0:
                    # This series represents the average performance of one seed across all envs
                    seed_summary_series = ScalarSeries(steps=grid, values=mean)
                    algo_to_seed_summary_series[algo].append(seed_summary_series)
        
        # Step 2: Aggregate the per-seed summary curves to get a final mean and std
        print("Step 2/2: Aggregating summary curves across all seeds...")
        fig_summary, ax_summary = plt.subplots(figsize=(6, 6), dpi=args.dpi)
        
        for algo, seed_summary_list in tqdm(algo_to_seed_summary_series.items()):
            if not seed_summary_list: continue
            
            # Aggregate the summary curves from all seeds for the final plot
            # No additional smoothing is applied here (smooth_alpha=0.0)
            final_grid, final_mean, final_std = aggregate_series(seed_summary_list, args.points, smooth_alpha=0.0)
            
            if final_grid.size > 0:
                ax_summary.plot(final_grid, final_mean, label=algo, color=color_map[algo])
                ax_summary.fill_between(final_grid, final_mean - final_std, final_mean + final_std,
                                        alpha=0.2, color=color_map[algo], linewidth=0)

        ax_summary.set_title("Overall Performance Summary" if args.title is None else args.title)
        ax_summary.set_ylabel("Mean Episodic Return (across all envs)")
        ax_summary.set_xlabel("Aligned Steps")
        ax_summary.grid(True)
        ax_summary.spines['top'].set_visible(False)
        ax_summary.spines['right'].set_visible(False)
        ax_summary.xaxis.set_major_formatter(mticker.FuncFormatter(human_readable_formatter))
        ax_summary.yaxis.set_major_formatter(mticker.FuncFormatter(human_readable_formatter))
        # ax_summary.legend(fontsize=12)
        legend_handles = [plt.Line2D([0], [0], color=color_map[algo], lw=3, label=abbr_algo) for algo, abbr_algo in zip(all_algos_labels, all_algos_abbr_labels)]
        ax_summary.legend(handles=legend_handles)

        fig_summary.tight_layout()
        summary_path = os.path.join(args.out, "all_envs_summary.png")
        fig_summary.savefig(summary_path)
        plt.close(fig_summary)
        print(f"[OK] Saved summary figure with new logic: {summary_path}")


if __name__ == "__main__":
    main()