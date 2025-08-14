
#!/usr/bin/env python3
"""
Plot TensorBoard `charts/episodic_return` aggregated by seed, grouped by environment
(one figure per environment) and with one curve per algorithm. The mean is plotted
as a solid line and the standard deviation is shown as a translucent band.

Directory layout expected (example):
./logs/
├── sac/
│   ├── basic_continuous_mlp/
│   │   ├── Hopper-v4__gymnasium/
│   │   │   ├── seed_0_0/20250811-234258/tb/events.out.tfevents....
│   │   │   ├── seed_1_1/...
│   │   │   └── seed_2_2/...
│   └── ...
└── simba_continuous_mlp/
    └── ...

Usage:
  python ./demos/common/plot_tb_graphs.py --logdir ./logs --tag charts/episodic_return --out ./logs/plots --points 100 --smooth 0.5

Notes:
- Requires `tensorboard` Python package (for EventAccumulator). Install with: pip install tensorboard
- Uses only matplotlib (no seaborn; no custom colors).
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

# Try to import TensorBoard's event accumulator; give a clear error if missing.
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    raise SystemExit(
        "Failed to import TensorBoard's EventAccumulator. "
        "Install it via `pip install tensorboard`.\n"
        f"Original error: {e}"
    )

SEED_DIR_RE = re.compile(r"^seed_(\d+)_\d+$")

@dataclass
class ScalarSeries:
    steps: np.ndarray  # shape [N]
    values: np.ndarray  # shape [N]

def list_immediate_subdirs(path: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    except FileNotFoundError:
        return []

def find_tb_event_file(tb_dir: str) -> Optional[str]:
    if not os.path.isdir(tb_dir):
        return None
    # Find newest event file in tb_dir
    candidates = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.startswith("events.out.tfevents")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def pick_latest_timestamp_dir(seed_dir: str) -> Optional[str]:
    # seed_dir contains timestamp subdirs like 20250811-234258
    ts_dirs = [d for d in list_immediate_subdirs(seed_dir) if re.match(r"^\d{8}-\d{6}$", d)]
    if not ts_dirs:
        return None
    # Sort lexicographically = chronological for this format
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
        # Sort by step in case of out-of-order
        idx = np.argsort(steps)
        return ScalarSeries(steps=steps[idx], values=vals[idx])
    except Exception as e:
        print(f"[WARN] Failed reading {event_path}: {e}")
        return None

def ewma_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return y
    s = np.empty_like(y)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = alpha * y[i] + (1 - alpha) * s[i-1]
    return s

def aggregate_across_seeds(series_list: List[ScalarSeries], num_points: int, smooth_alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate all seed series onto a common step grid within the intersection
    of their step ranges, then compute mean and std across seeds.
    Returns (grid_steps, mean, std).
    """
    if not series_list:
        return np.array([]), np.array([]), np.array([])

    # Compute intersection of step ranges
    min_max = min(s.steps.max() for s in series_list if len(s.steps) > 0)
    max_min = max(s.steps.min() for s in series_list if len(s.steps) > 0)
    if not np.isfinite(min_max) or not np.isfinite(max_min) or min_max <= max_min:
        # Fallback to use any one series grid
        ref = series_list[0]
        grid = ref.steps
    else:
        grid = np.linspace(max_min, min_max, num_points)

    # Interpolate each series
    aligned = []
    for s in series_list:
        if len(s.steps) < 2:
            # Not enough points to interpolate—skip
            continue
        vals = np.interp(grid, s.steps, s.values)
        vals = ewma_smooth(vals, smooth_alpha)
        aligned.append(vals)

    if not aligned:
        return np.array([]), np.array([]), np.array([])

    A = np.vstack(aligned)  # [S, T]
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
    parser.add_argument("--avail-algos", nargs='+', default=['sac'])
    parser.add_argument("--avail-envs", nargs='+', default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    algos = list_immediate_subdirs(args.logdir)  # e.g., sac, simba_continuous_mlp
    algos = [a for a in algos if a in args.avail_algos]  # Filter by available algorithms
    if not algos:
        print(f"[ERROR] No algorithm folders under {args.logdir}.")
        return

    # env_to_algo_series: env -> algo -> list[ScalarSeries]
    env_to_algo_series: Dict[str, Dict[str, List[ScalarSeries]]] = {}

    env_algo_fam_paths = []

    # find all seed paths
    for algo in algos:
        algo_root = os.path.join(args.logdir, algo)
        families = list_immediate_subdirs(algo_root)  # e.g., basic_continuous_mlp
        for fam in families:
            fam_root = os.path.join(algo_root, fam)
            envs = list_immediate_subdirs(fam_root)  # e.g., Hopper-v4__gymnasium
            for env in envs:
                if args.avail_envs and env not in args.avail_envs:
                    continue
                env_root = os.path.join(fam_root, env)
                seed_dirs = [d for d in list_immediate_subdirs(env_root) if SEED_DIR_RE.match(d)]
                for seed_dir in seed_dirs:
                    seed_path = os.path.join(env_root, seed_dir)
                    env_algo_fam_paths.append([env, algo, fam, seed_path])
    
    for env, algo, fam, seed_path in tqdm(env_algo_fam_paths):
        ts_dir = pick_latest_timestamp_dir(seed_path)
        if ts_dir is None:
            print(f"[WARN] No timestamp subdir under {seed_path}")
            continue
        tb_dir = os.path.join(ts_dir, "tb")
        event_file = find_tb_event_file(tb_dir)
        if event_file is None:
            print(f"[WARN] No event file under {tb_dir}")
            continue
        series = load_scalar_series_from_event_file(event_file, args.tag)
        if series is None:
            print(f"[WARN] Tag '{args.tag}' not found in {event_file}")
            continue
        env_to_algo_series.setdefault(env, {}).setdefault(f"{algo}_{fam}", []).append(series)

    if not env_to_algo_series:
        print("[ERROR] No data found. Check --logdir and --tag.")
        return

    # Plot per environment
    for env, algo_map in sorted(env_to_algo_series.items()):
        plt.figure(figsize=(7, 4.5), dpi=args.dpi)
        any_plotted = False
        for algo, series_list in sorted(algo_map.items()):
            grid, mean, std = aggregate_across_seeds(series_list, args.points, args.smooth)
            if grid.size == 0:
                print(f"[WARN] Not enough data to plot {env} / {algo}")
                continue
            # Mean line
            plt.plot(grid, mean, label=algo, linewidth=1.8)
            # Variance band (std^2). Visualize via ±std shading.
            plt.fill_between(grid, mean - std, mean + std, alpha=0.25, linewidth=0)
            any_plotted = True

        if not any_plotted:
            plt.close()
            print(f"[WARN] Nothing plotted for env {env}; skipping figure.")
            continue

        plt.title(env)
        plt.xlabel("Env Step")
        plt.ylabel("Episodic Return")
        plt.legend(loc="best", frameon=True)
        plt.grid(True, linewidth=0.4, alpha=0.5)

        # sanitize filename
        safe_env = re.sub(r"[^a-zA-Z0-9_.\-]+", "_", env)
        out_path_png = os.path.join(args.out, f"{safe_env}.png")
        # out_path_svg = os.path.join(args.out, f"{safe_env}.svg")
        plt.tight_layout()
        plt.savefig(out_path_png, bbox_inches="tight")
        # plt.savefig(out_path_svg, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved {out_path_png}")

    print("[DONE] All single figures saved.")

    env_list = sorted(env_to_algo_series.keys())
    n_envs = len(env_list)
    ncols = 2
    nrows = math.ceil(n_envs / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), dpi=args.dpi)
    axes = np.array(axes).reshape(-1)

    for ax_idx, env in enumerate(env_list):
        ax = axes[ax_idx]
        algo_map = env_to_algo_series[env]
        any_plotted = False
        for algo, series_list in sorted(algo_map.items()):
            grid, mean, std = aggregate_across_seeds(series_list, args.points, args.smooth)
            if grid.size == 0:
                print(f"[WARN] Not enough data to plot {env} / {algo}")
                continue
            ax.plot(grid, mean, label=algo, linewidth=1.8)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.25, linewidth=0)
            any_plotted = True
        if any_plotted:
            ax.set_title(env)
            ax.set_xlabel("Step")
            ax.set_ylabel("Episodic Return")
            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.legend(loc="best", frameon=True)
        else:
            ax.set_visible(False)  # 没有数据就隐藏subplot

    # 如果subplot数量多于环境数量，隐藏多余的空subplot
    for ax in axes[len(env_list):]:
        ax.set_visible(False)

    plt.tight_layout()
    overview_path = os.path.join(args.out, "all_envs_overview.png")
    plt.savefig(overview_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved overview figure: {overview_path}")

if __name__ == "__main__":
    main()
