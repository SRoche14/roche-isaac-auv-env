#!/usr/bin/env python3
"""Plot drag force/torque logs in MITRE/LTurn.

Generates two figures per log:
- Linear velocity over time + drag·velocity
- Angular velocity over time + torque·ang_vel

If per-NN outputs are available, it also generates the same pair of plots
for each NN output (linear and angular).
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

COMPONENT_LABELS = ("x", "y", "z")
COMPONENT_COLORS = ("tab:red", "tab:green", "tab:blue")
NN_SPECS = (
    ("linear", "Linear NN", "drag_force_linear_b", "drag_torque_linear_b"),
    ("angular", "Angular NN", "drag_force_angular_b", "drag_torque_angular_b"),
)


def _load_records(path: Path) -> List[dict]:
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data:
        records = data["records"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unexpected drag log format in {path}")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No records found in {path}")
    return records


def _series_from_records(records: List[dict], key: str) -> Optional[np.ndarray]:
    if key not in records[0]:
        return None
    values = [r.get(key) for r in records]
    arr = np.array(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected Nx3 data for {key}")
    return arr


def _series_sum(records: List[dict], keys: Iterable[str]) -> Optional[np.ndarray]:
    arrays = []
    for key in keys:
        series = _series_from_records(records, key)
        if series is None:
            return None
        arrays.append(series)
    return np.sum(arrays, axis=0)


def _extract_times(records: List[dict]) -> np.ndarray:
    if "t" in records[0]:
        times = np.array([r.get("t", 0.0) for r in records], dtype=float)
    else:
        times = np.arange(len(records), dtype=float)
    return times


def _extract_drag_force(records: List[dict]) -> np.ndarray:
    force = _series_from_records(records, "drag_force_b")
    if force is not None:
        return force
    summed = _series_sum(records, ("drag_force_linear_b", "drag_force_angular_b"))
    if summed is not None:
        return summed
    raise ValueError("No drag force series found")


def _extract_drag_torque(records: List[dict]) -> np.ndarray:
    torque = _series_from_records(records, "drag_torque_b")
    if torque is not None:
        return torque
    summed = _series_sum(records, ("drag_torque_linear_b", "drag_torque_angular_b"))
    if summed is not None:
        return summed
    raise ValueError("No drag torque series found")


def _extract_nn_outputs(
    records: List[dict],
    force_key: str,
    torque_key: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    force = _series_from_records(records, force_key)
    torque = _series_from_records(records, torque_key)
    if force is None or torque is None:
        return None
    return force, torque


def _plot_components(
    ax: plt.Axes,
    times: np.ndarray,
    values: np.ndarray,
    labels: Tuple[str, str, str],
    ylabel: str,
) -> None:
    for idx, label in enumerate(labels):
        ax.plot(
            times,
            values[:, idx],
            label=label,
            color=COMPONENT_COLORS[idx],
            linewidth=1.0,
        )
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=3, fontsize=8)


def _plot_dot_product(
    ax: plt.Axes,
    times: np.ndarray,
    velocity: np.ndarray,
    drag: np.ndarray,
    ylabel: str,
) -> None:
    dot = np.sum(velocity * drag, axis=1)
    ax.plot(times, dot, color="tab:purple", linewidth=1.0)
    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_force_and_velocity(
    name: str,
    times: np.ndarray,
    forces: np.ndarray,
    lin_vel: np.ndarray,
    title_prefix: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6.5))
    prefix = f"{title_prefix} " if title_prefix else ""
    axes[0].set_title(f"{name} - {prefix}Linear Velocity and Drag·Velocity")
    _plot_components(
        axes[0],
        times,
        lin_vel,
        ("vx", "vy", "vz"),
        "Linear velocity [m/s]",
    )
    _plot_dot_product(
        axes[1],
        times,
        lin_vel,
        forces,
        "Drag·Velocity [N*m/s]",
    )
    axes[1].set_xlabel("Time [s]")
    fig.tight_layout()
    return fig


def plot_torque_and_ang_velocity(
    name: str,
    times: np.ndarray,
    torques: np.ndarray,
    ang_vel: np.ndarray,
    title_prefix: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6.5))
    prefix = f"{title_prefix} " if title_prefix else ""
    axes[0].set_title(f"{name} - {prefix}Angular Velocity and Torque·AngVel")
    _plot_components(
        axes[0],
        times,
        ang_vel,
        ("wx", "wy", "wz"),
        "Angular velocity [rad/s]",
    )
    _plot_dot_product(
        axes[1],
        times,
        ang_vel,
        torques,
        "Torque·AngVel [N*m*rad/s]",
    )
    axes[1].set_xlabel("Time [s]")
    fig.tight_layout()
    return fig


def collect_drag_logs(base_dir: Path, pattern: str) -> List[Path]:
    return sorted(base_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("MITRE/LTurn"),
        help="Directory containing drag log JSON files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="drag_forces_moments_*.json",
        help="Glob pattern for drag log files.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save PNGs; if omitted, plots are only displayed.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive plots (useful with --save-dir).",
    )
    parser.add_argument(
        "--no-nn",
        action="store_true",
        help="Skip per-NN plots even if NN outputs exist.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    logs = collect_drag_logs(base_dir, args.pattern)
    if not logs:
        raise SystemExit(f"No drag logs found in {base_dir} matching '{args.pattern}'")

    show_plots = not args.no_show
    for path in logs:
        try:
            records = _load_records(path)
            times = _extract_times(records)
            forces = _extract_drag_force(records)
            torques = _extract_drag_torque(records)
            lin_vel = _series_from_records(records, "lin_vel_b")
            ang_vel = _series_from_records(records, "ang_vel_b")
            if lin_vel is None or ang_vel is None:
                raise ValueError("Missing velocity series")
        except Exception as exc:
            print(f"[WARN] Skipping {path}: {exc}")
            continue

        name = path.stem
        fig_force = plot_force_and_velocity(name, times, forces, lin_vel)
        fig_torque = plot_torque_and_ang_velocity(name, times, torques, ang_vel)
        nn_figs: List[Tuple[str, plt.Figure, plt.Figure]] = []
        if not args.no_nn:
            for slug, label, force_key, torque_key in NN_SPECS:
                nn_outputs = _extract_nn_outputs(records, force_key, torque_key)
                if nn_outputs is None:
                    continue
                nn_force, nn_torque = nn_outputs
                nn_force_fig = plot_force_and_velocity(
                    name,
                    times,
                    nn_force,
                    lin_vel,
                    title_prefix=label,
                )
                nn_torque_fig = plot_torque_and_ang_velocity(
                    name,
                    times,
                    nn_torque,
                    ang_vel,
                    title_prefix=label,
                )
                nn_figs.append((slug, nn_force_fig, nn_torque_fig))

        if args.save_dir:
            force_path = args.save_dir / f"{name}_force_linvel.png"
            torque_path = args.save_dir / f"{name}_torque_angvel.png"
            fig_force.savefig(force_path, dpi=150)
            fig_torque.savefig(torque_path, dpi=150)
            for slug, nn_force_fig, nn_torque_fig in nn_figs:
                nn_force_path = args.save_dir / f"{name}_{slug}_force_linvel.png"
                nn_torque_path = args.save_dir / f"{name}_{slug}_torque_angvel.png"
                nn_force_fig.savefig(nn_force_path, dpi=150)
                nn_torque_fig.savefig(nn_torque_path, dpi=150)

        if not show_plots:
            plt.close(fig_force)
            plt.close(fig_torque)
            for _, nn_force_fig, nn_torque_fig in nn_figs:
                plt.close(nn_force_fig)
                plt.close(nn_torque_fig)

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
