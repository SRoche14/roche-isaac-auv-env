#!/usr/bin/env python3
"""
Plot velocity profiles for all Qualisys Curee Test JSON files across MITRE trajectories.

Skips tests excluded in trajectory plotting scripts:
- Left: {7}
- Up: {7, 12}
- LTurn: {7, 12, 17, 21}
- Circle: {7, 12}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PREFERRED_MARKER = "CUREE - 1"
FALLBACK_MARKER_PREFIX = "CUREE"

TRAJECTORIES = (
    {"label": "Forward", "base_dir": Path("MITRE/Forward"), "excluded_tests": set()},
    {"label": "Left", "base_dir": Path("MITRE/Left"), "excluded_tests": {7}},
    {"label": "Up", "base_dir": Path("MITRE/Up"), "excluded_tests": {7, 12}},
    {"label": "LTurn", "base_dir": Path("MITRE/LTurn"), "excluded_tests": {7, 12, 17, 21}},
    {"label": "Circle", "base_dir": Path("MITRE/Circles"), "excluded_tests": {7, 12}},
)


def _part_length(part: Dict) -> int:
    range_info = part.get("Range") or {}
    start = range_info.get("Start")
    end = range_info.get("End")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return end - start + 1
    values = part.get("Values", [])
    return len(values) if isinstance(values, list) else 0


def _contiguous_lengths(parts: List[Dict]) -> Tuple[int, int]:
    best = 0
    total = 0
    for part in parts:
        length = _part_length(part)
        if length > best:
            best = length
        total += length
    return best, total


def _best_entry(entries: List[Dict], name_prefix: Optional[str] = None) -> Optional[Dict]:
    candidates = []
    prefix = name_prefix.upper() if name_prefix else None
    for entry in entries:
        name = entry.get("Name", "")
        if prefix and not name.upper().startswith(prefix):
            continue
        best, total = _contiguous_lengths(entry.get("Parts", []))
        if best <= 0:
            continue
        candidates.append((best, total, entry))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _has_values(entry: Dict) -> bool:
    best, _ = _contiguous_lengths(entry.get("Parts", []))
    return best > 0


def _find_marker_by_name(markers: List[Dict], name: str) -> Optional[Dict]:
    for marker in markers:
        if marker.get("Name") == name:
            return marker
    return None


def _marker_positions(marker: Dict) -> Optional[np.ndarray]:
    parts = marker.get("Parts", [])
    chunks = []
    for part in parts:
        vals = np.array(part.get("Values", []), dtype=float)
        if vals.size == 0:
            continue
        if vals.ndim != 2 or vals.shape[1] < 3:
            raise ValueError(f"Unexpected Values shape for '{marker.get('Name', '<unknown>')}': {vals.shape}")
        chunks.append(vals[:, :3])
    if not chunks:
        return None
    return np.vstack(chunks) / 1000.0


def _rigid_body_sample_xyz(sample: object) -> Optional[List[float]]:
    if isinstance(sample, (list, tuple)) and sample:
        head = sample[0]
        if isinstance(head, (list, tuple)) and len(head) >= 3:
            return list(head[:3])
        if len(sample) >= 3 and all(isinstance(x, (int, float)) for x in sample[:3]):
            return list(sample[:3])
    return None


def _rigid_body_positions(rigid_body: Dict) -> Optional[np.ndarray]:
    parts = rigid_body.get("Parts", [])
    positions = []
    for part in parts:
        values = part.get("Values", [])
        if not isinstance(values, list):
            continue
        for sample in values:
            xyz = _rigid_body_sample_xyz(sample)
            if xyz is None:
                continue
            positions.append(xyz)
    if not positions:
        return None
    return np.array(positions, dtype=float) / 1000.0


def _select_ground_truth_entry(data: Dict) -> Tuple[str, Dict]:
    markers = data.get("Markers", [])
    rigid_bodies = data.get("RigidBodies", [])
    preferred = _find_marker_by_name(markers, PREFERRED_MARKER)
    if preferred is not None and _has_values(preferred):
        return "marker", preferred

    curee_candidates = [
        marker
        for marker in markers
        if marker.get("Name", "").upper().startswith(FALLBACK_MARKER_PREFIX)
        and marker.get("Name") != PREFERRED_MARKER
    ]
    curee_marker = _best_entry(curee_candidates)
    if curee_marker is not None:
        return "marker", curee_marker

    best_marker = _best_entry(markers)
    if best_marker is not None:
        return "marker", best_marker

    best_rigid = _best_entry(rigid_bodies)
    if best_rigid is not None:
        return "rigid_body", best_rigid
    raise ValueError("No marker or rigid body data found in the Qualisys file")


def _extract_frequency(data: Dict, entry: Dict) -> float:
    for timebase in (data.get("Timebase"), entry.get("Timebase")):
        if isinstance(timebase, dict):
            freq = timebase.get("Frequency")
            try:
                return float(freq)
            except (TypeError, ValueError):
                pass
    return 100.0


def load_positions(path: Path) -> Tuple[str, str, np.ndarray, float]:
    with path.open("r") as f:
        data = json.load(f)
    source_type, entry = _select_ground_truth_entry(data)
    if source_type == "marker":
        positions = _marker_positions(entry)
    else:
        positions = _rigid_body_positions(entry)
    if positions is None:
        raise ValueError("Selected entry has no positions")
    freq = _extract_frequency(data, entry)
    name = entry.get("Name", "<unknown>")
    return source_type, name, positions, freq


def mad(arr: np.ndarray) -> float:
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def spike_smoothing(
    traj: np.ndarray,
    freq_hz: float = 100.0,
    process_acc_var: float = 0.05,
    meas_var_floor: float = 1e-3,
    gate_thresh: float = 2.0,
) -> np.ndarray:
    """
    Robust constant-velocity Kalman filter with RTS smoothing.
    - Measurement outliers are down-weighted via Mahalanobis gating (gate_thresh).
    - Process noise is set by process_acc_var (acceleration variance).
    - Output length matches input length to preserve timing.
    """
    n = len(traj)
    if n < 3:
        return traj

    dt = 1.0 / float(freq_hz)
    dim = 3
    state_dim = 2 * dim

    F = np.eye(state_dim)
    F[:dim, dim:] = np.eye(dim) * dt

    q = process_acc_var
    Q_pos = (dt ** 4) / 4.0 * q * np.eye(dim)
    Q_cross = (dt ** 3) / 2.0 * q * np.eye(dim)
    Q_vel = (dt ** 2) * q * np.eye(dim)
    Q = np.block([[Q_pos, Q_cross], [Q_cross, Q_vel]])

    H = np.hstack([np.eye(dim), np.zeros((dim, dim))])

    step = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    mad_step = mad(step) if len(step) > 0 else 0.0
    r_base = max(mad_step ** 2, meas_var_floor)
    R = np.eye(dim) * r_base

    x = np.zeros((state_dim,))
    x[:dim] = traj[0]
    P = np.eye(state_dim)

    xs = np.zeros((n, state_dim))
    Ps = np.zeros((n, state_dim, state_dim))

    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q

        z = traj[k]
        y = z - H @ x
        S = H @ P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        mahal = float(y.T @ S_inv @ y)
        if mahal > gate_thresh:
            scale = mahal / gate_thresh
            S = S + (scale - 1.0) * R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

        K = P @ H.T @ S_inv
        x = x + K @ y
        P = (np.eye(state_dim) - K @ H) @ P

        xs[k] = x
        Ps[k] = P

    x_smooth = xs.copy()
    P_smooth = Ps.copy()
    for k in range(n - 2, -1, -1):
        P_pred = F @ Ps[k] @ F.T + Q
        try:
            Ck = Ps[k] @ F.T @ np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            Ck = Ps[k] @ F.T @ np.linalg.pinv(P_pred)
        x_smooth[k] = xs[k] + Ck @ (x_smooth[k + 1] - F @ xs[k])
        P_smooth[k] = Ps[k] + Ck @ (P_smooth[k + 1] - P_pred) @ Ck.T

    return x_smooth[:, :dim]


def compute_velocity(traj: np.ndarray, freq_hz: float) -> np.ndarray:
    if traj.shape[0] < 2:
        return np.empty((0, 3), dtype=float)
    return np.diff(traj, axis=0) * float(freq_hz)


def parse_test_id(path: Path) -> Optional[int]:
    match = re.search(r"Curee Test(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def collect_tests() -> List[Tuple[str, Path, int]]:
    collected = []
    for traj in TRAJECTORIES:
        base_dir = traj["base_dir"]
        if not base_dir.exists():
            print(f"[WARN] Missing directory: {base_dir}")
            continue
        excluded = traj["excluded_tests"]
        tests = []
        for path in base_dir.glob("Curee Test*.json"):
            test_id = parse_test_id(path)
            if test_id is None:
                continue
            if test_id in excluded:
                continue
            tests.append((path, test_id))
        for path, test_id in sorted(tests, key=lambda item: item[1]):
            collected.append((traj["label"], path, test_id))
    return collected


def plot_velocity(traj_label: str, test_id: int, velocity: np.ndarray, freq_hz: float) -> plt.Figure:
    t = np.arange(velocity.shape[0]) / float(freq_hz)
    speed = np.linalg.norm(velocity, axis=1)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(t, velocity[:, 0], label="vx")
    axes[0].plot(t, velocity[:, 1], label="vy")
    axes[0].plot(t, velocity[:, 2], label="vz")
    axes[0].set_ylabel("Velocity [m/s]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, speed, color="black", label="speed")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Speed [m/s]")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f"{traj_label} Test{test_id:04d} Velocity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot velocity profiles for all Curee Test JSON files in MITRE."
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
    args = parser.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    tests = collect_tests()
    if not tests:
        print("[WARN] No Curee Test JSON files found.")
        return

    show_plots = not args.no_show
    for traj_label, path, test_id in tests:
        try:
            source_type, source_name, positions, freq_hz = load_positions(path)
        except Exception as exc:
            print(f"[WARN] Skipping {path}: {exc}")
            continue

        positions_smooth = spike_smoothing(positions, freq_hz=freq_hz)
        velocity = compute_velocity(positions_smooth, freq_hz)
        if velocity.size == 0:
            print(f"[WARN] {path} has fewer than 2 samples; skipping.")
            continue

        print(
            f"[INFO] {traj_label} Test{test_id:04d}: {source_type} '{source_name}', "
            f"{len(positions)} samples @ {freq_hz:.1f} Hz"
        )
        fig = plot_velocity(traj_label, test_id, velocity, freq_hz)
        if args.save_dir:
            safe_label = traj_label.replace(" ", "")
            out_path = args.save_dir / f"{safe_label}_Test{test_id:04d}_velocity.png"
            fig.savefig(out_path, dpi=150)
        if not show_plots:
            plt.close(fig)

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
