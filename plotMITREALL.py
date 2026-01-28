#!/usr/bin/env python3
"""
Aggregate MITRE runs across all directions and compare CFD vs Cuboid.

Creates box plots for position RMSE, velocity RMSE, and Frechet distance for:
1) Raw GT
2) Smoothed GT
3) Smoothed GT after a 2-second cut
and runs paired t-tests between CFD and Cuboid for each metric set.

Alignment uses a yaw-only fit on the first 1.5 meters of motion for each trajectory
relative to GT, and metrics are evaluated after that first meter segment.
"""

import json
import re
from math import erf, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    {
        "name": "Forward",
        "base_dir": Path("MITRE/Forward"),
        "bag_prefix": "mitreForward",
        "odom_prefix": "odomMITREForward",
        "cfd_prefix": "positionsMITREForwardCFD",
        "cuboid_prefix": "positionsMITREForwardCuboid",
        "qualy_prefix": "qualyForward",
        "mitre_prefix": "mitreForward",
        "start_times": "startingForwardTimes.json",
        "axis_mode": "xy",
        "exclude_tests": {7, 12},
    },
    {
        "name": "Left",
        "base_dir": Path("MITRE/Left"),
        "bag_prefix": "mitreLeft",
        "odom_prefix": "odomMITRELeft",
        "cfd_prefix": "positionsMITRELeftCFD",
        "cuboid_prefix": "positionsMITRELeftCuboid",
        "qualy_prefix": "qualyLeft",
        "mitre_prefix": "mitreLeft",
        "start_times": "startingLeftTimes.json",
        "axis_mode": "xy",
        "exclude_tests": {7, 12},
    },
    {
        "name": "Up",
        "base_dir": Path("MITRE/Up"),
        "bag_prefix": "mitreUp",
        "odom_prefix": "odomMITREUp",
        "cfd_prefix": "positionsMITREUpCFD",
        "cuboid_prefix": "positionsMITREUpCuboid",
        "qualy_prefix": "qualyUp",
        "mitre_prefix": "mitreUp",
        "start_times": "startingUpTimes.json",
        "axis_mode": "z",
        "exclude_tests": {7, 12},
    },
    {
        "name": "LTurn",
        "base_dir": Path("MITRE/LTurn"),
        "bag_prefix": "mitreLTurn",
        "odom_prefix": "odomMITRELTurn",
        "cfd_prefix": "positionsMITRELTurnCFD",
        "cuboid_prefix": "positionsMITRELTurnCuboid",
        "qualy_prefix": "qualyLTurn",
        "mitre_prefix": "mitreLTurn",
        "start_times": "startingLTurnTimes.json",
        "axis_mode": "xy",
        "exclude_tests": {7, 12, 17, 21},
        "pre_trim_seconds": 0.0,
    },
    {
        "name": "Circles",
        "base_dir": Path("MITRE/Circles"),
        "bag_prefix": "mitreCircle",
        "odom_prefix": "odomMITRECircle",
        "cfd_prefix": "positionsMITRECircleCFD",
        "cuboid_prefix": "positionsMITRECircleCuboid",
        "qualy_prefix": "qualyCircle",
        "mitre_prefix": "mitreCircle",
        "start_times": "startingCirclesTimes.json",
        "axis_mode": "xy",
        "exclude_tests": {7, 12},
    },
]

METRIC_LABELS = [
    ("rmse_pos", "RMSE position (m)"),
    ("vel_rmse", "Velocity RMSE (m/s)"),
    ("frechet", "Frechet distance (m)"),
]

ALIGN_DISTANCE_M = 1.5


def load_xyz_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


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


def _marker_positions(marker: Dict) -> Optional[np.ndarray]:
    name = marker.get("Name", "<unknown>")
    parts = marker.get("Parts", [])
    chunks = []
    for part in parts:
        vals = np.array(part.get("Values", []), dtype=float)
        if vals.size == 0:
            continue
        if vals.ndim != 2 or vals.shape[1] < 3:
            raise ValueError(f"Unexpected Values shape for '{name}': {vals.shape}")
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


def _rigid_body_positions(body: Dict) -> Optional[np.ndarray]:
    parts = body.get("Parts", [])
    chunks = []
    for part in parts:
        values = part.get("Values", [])
        if not values:
            continue
        sample = _rigid_body_sample_xyz(values[0])
        if sample is None:
            continue
        chunk = np.array([_rigid_body_sample_xyz(v) for v in values], dtype=float)
        if chunk.ndim != 2 or chunk.shape[1] < 3:
            continue
        chunks.append(chunk[:, :3])
    if not chunks:
        return None
    return np.vstack(chunks) / 1000.0


def _select_ground_truth_entry(data: Dict) -> Tuple[str, Dict]:
    markers = data.get("Markers", [])
    rigid_bodies = data.get("RigidBodies", [])
    marker = _best_entry(markers, name_prefix="CUREE") or _best_entry(markers)
    rigid = _best_entry(rigid_bodies)
    if marker is None and rigid is None:
        raise ValueError("No marker or rigid body data found in the Qualisys file")
    if marker is None:
        return "rigid", rigid
    if rigid is None:
        return "marker", marker
    m_best, _ = _contiguous_lengths(marker.get("Parts", []))
    r_best, _ = _contiguous_lengths(rigid.get("Parts", []))
    return ("marker", marker) if m_best >= r_best else ("rigid", rigid)


def _parts_sample_span(parts: List[Dict]) -> Optional[Tuple[int, int]]:
    min_start = None
    max_end = None
    for part in parts:
        range_info = part.get("Range") or {}
        start = range_info.get("Start")
        end = range_info.get("End")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        min_start = start if min_start is None else min(min_start, start)
        max_end = end if max_end is None else max(max_end, end)
    if min_start is None or max_end is None:
        return None
    return min_start, max_end


def load_ground_truth(path: Path) -> np.ndarray:
    with path.open("r") as f:
        data = json.load(f)
    source_type, entry = _select_ground_truth_entry(data)
    if source_type == "marker":
        positions = _marker_positions(entry)
    else:
        positions = _rigid_body_positions(entry)
    if positions is None:
        raise ValueError("Selected ground truth entry has no Values in Parts")
    return positions


def ground_truth_duration_seconds(path: Path) -> float:
    with path.open("r") as f:
        data = json.load(f)
    _, entry = _select_ground_truth_entry(data)
    span = _parts_sample_span(entry.get("Parts", []))
    if span is None:
        return 0.0
    min_start, max_end = span
    return max(0.0, (max_end - min_start) / 100.0)


def load_start_times(path: Path, prefix: str) -> Dict[int, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing start times file: {path}")
    with path.open("r") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith(prefix):
            run_id = int(k.replace(prefix, ""))
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(val):
                continue
            out[run_id] = val
    return out


def trim_by_samples(traj: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(traj) == 0:
        return traj
    if n >= len(traj):
        return traj[-1:].copy()
    return traj[n:]


def trim_by_time(traj: np.ndarray, freq_hz: float, start_time: float) -> np.ndarray:
    start_idx = int(start_time * freq_hz)
    return trim_by_samples(traj, start_idx)


def limit_by_time(traj: np.ndarray, freq_hz: float, max_seconds: float) -> np.ndarray:
    if len(traj) == 0:
        return traj
    if max_seconds <= 0:
        return traj[:1]
    n = int(max_seconds * freq_hz)
    if n < 2:
        n = 2
    return traj[: min(len(traj), n)]


def umeyama_align(
    source: np.ndarray,
    target: np.ndarray,
    fit_target: Optional[np.ndarray] = None,
    max_seconds: float = 6.0,
    freq_source: float = 100.0,
    freq_target: float = 100.0,
) -> np.ndarray:
    if len(source) == 0 or len(target) == 0:
        return source
    tgt_full = fit_target if fit_target is not None else target
    src_fit = limit_by_time(source, freq_source, max_seconds)
    tgt_fit = limit_by_time(tgt_full, freq_target, max_seconds)
    if freq_target > freq_source and len(tgt_fit) > len(src_fit):
        step = int(round(freq_target / freq_source))
        step = max(1, step)
        tgt_fit = tgt_fit[::step]
    elif freq_source > freq_target and len(src_fit) > len(tgt_fit):
        step = int(round(freq_source / freq_target))
        step = max(1, step)
        src_fit = src_fit[::step]
    n = min(len(src_fit), len(tgt_fit))
    A0 = src_fit[:n] - src_fit[0]
    B0 = tgt_fit[:n] - tgt_fit[0]

    A_xy = A0[:, :2]
    B_xy = B0[:, :2]
    cov = (B_xy.T @ A_xy) / n
    U, _, Vt = np.linalg.svd(cov)
    S2 = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S2[-1, -1] = -1
    R2 = U @ S2 @ Vt
    R = np.eye(3)
    R[:2, :2] = R2
    t = target[0] - R @ source[0]
    return (R @ source.T).T + t


def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def spike_smoothing(
    traj: np.ndarray,
    freq_hz: float = 100.0,
    process_acc_var: float = 0.05,
    meas_var_floor: float = 1e-3,
    gate_thresh: float = 2.0,
) -> np.ndarray:
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
    if len(traj) < 2:
        return np.empty((0, 3))
    return np.diff(traj, axis=0) * freq_hz


def resample_to_times(traj: np.ndarray, freq_hz: float, times: np.ndarray) -> np.ndarray:
    if len(traj) == 0 or len(times) == 0:
        return np.empty((0, 3))
    t_in = np.arange(len(traj)) / freq_hz
    out = np.zeros((len(times), traj.shape[1]))
    for d in range(traj.shape[1]):
        out[:, d] = np.interp(times, t_in, traj[:, d])
    return out


def build_common_time_grid(len_a: int, len_b: int, freq_a: float, freq_b: float) -> Tuple[np.ndarray, float]:
    if len_a == 0 or len_b == 0:
        return np.array([]), 0.0
    dur_a = (len_a - 1) / freq_a if len_a > 1 else 0.0
    dur_b = (len_b - 1) / freq_b if len_b > 1 else 0.0
    duration = min(dur_a, dur_b)
    if duration <= 0:
        return np.array([]), 0.0
    freq_ref = min(freq_a, freq_b)
    n = int(np.floor(duration * freq_ref)) + 1
    times = np.arange(n) / freq_ref
    return times, freq_ref


def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float("nan")
    ca = np.full((n, m), np.inf)
    for i in range(n):
        for j in range(m):
            dist = np.linalg.norm(P[i] - Q[j])
            if i == 0 and j == 0:
                ca[i, j] = dist
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], dist)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], dist)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), dist)
    return float(ca[-1, -1])


def compute_metrics(truth: np.ndarray, other: np.ndarray, freq_truth: float, freq_other: float) -> Dict[str, float]:
    times, freq_ref = build_common_time_grid(len(truth), len(other), freq_truth, freq_other)
    if times.size == 0:
        return {k: float("nan") for k, _ in METRIC_LABELS}
    t_interp = resample_to_times(truth, freq_truth, times)
    o_interp = resample_to_times(other, freq_other, times)
    diff = o_interp - t_interp
    pos_err = np.linalg.norm(diff, axis=1)
    rmse_pos = float(np.sqrt(np.mean(pos_err ** 2)))
    tv = compute_velocity(t_interp, freq_ref)
    ov = compute_velocity(o_interp, freq_ref)
    m = min(len(tv), len(ov))
    if m > 0:
        vel_diff = ov[:m] - tv[:m]
        vel_err = np.linalg.norm(vel_diff, axis=1)
        vel_rmse = float(np.sqrt(np.mean(vel_err ** 2)))
    else:
        vel_rmse = float("nan")
    frechet = discrete_frechet(t_interp, o_interp)
    return {"rmse_pos": rmse_pos, "vel_rmse": vel_rmse, "frechet": frechet}


def _axis_displacement(traj: np.ndarray, axis_mode: str) -> np.ndarray:
    if len(traj) == 0:
        return np.array([])
    delta = traj - traj[0]
    if axis_mode == "z":
        return np.abs(delta[:, 2])
    return np.maximum(np.abs(delta[:, 0]), np.abs(delta[:, 1]))


def first_meter_index(traj: np.ndarray, axis_mode: str, threshold_m: float = ALIGN_DISTANCE_M) -> int:
    if len(traj) == 0:
        return 0
    disp = _axis_displacement(traj, axis_mode)
    hits = np.where(disp >= threshold_m)[0]
    return int(hits[0]) if hits.size else len(traj) - 1


def segment_until_first_meter(traj: np.ndarray, axis_mode: str) -> np.ndarray:
    if len(traj) == 0:
        return traj
    idx = first_meter_index(traj, axis_mode)
    return traj[: idx + 1]


def yaw_rotation_from_segments(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 2 or len(target) < 2:
        return np.eye(3)
    n = min(len(source), len(target))
    A0 = source[:n] - source[0]
    B0 = target[:n] - target[0]
    A_xy = A0[:, :2]
    B_xy = B0[:, :2]
    cov = (B_xy.T @ A_xy) / n
    U, _, Vt = np.linalg.svd(cov)
    S2 = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S2[-1, -1] = -1
    R2 = U @ S2 @ Vt
    R = np.eye(3)
    R[:2, :2] = R2
    return R


def compute_alignment_rotation(truth: np.ndarray, odom: np.ndarray, axis_mode: str) -> np.ndarray:
    truth_seg = segment_until_first_meter(truth, axis_mode)
    odom_seg = segment_until_first_meter(odom, axis_mode)
    times, _ = build_common_time_grid(len(truth_seg), len(odom_seg), 100.0, 50.0)
    if times.size == 0:
        return np.eye(3)
    truth_rs = resample_to_times(truth_seg, 100.0, times)
    odom_rs = resample_to_times(odom_seg, 50.0, times)
    return yaw_rotation_from_segments(odom_rs, truth_rs)


def apply_rotation_with_translation(traj: np.ndarray, rotation: np.ndarray, target_start: np.ndarray) -> np.ndarray:
    if len(traj) == 0:
        return traj
    return (rotation @ traj.T).T + (target_start - rotation @ traj[0])


def trim_after_first_meter(
    truth: np.ndarray,
    other: np.ndarray,
    axis_mode: str,
    freq_truth: float,
    freq_other: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(truth) == 0:
        return truth, other
    idx = first_meter_index(truth, axis_mode)
    cut_time = (idx + 1) / freq_truth
    return (
        trim_by_time(truth, freq_truth, cut_time),
        trim_by_time(other, freq_other, cut_time),
    )


def compute_metrics_after_meter(
    truth: np.ndarray,
    other: np.ndarray,
    axis_mode: str,
    freq_truth: float,
    freq_other: float,
) -> Dict[str, float]:
    truth_eval, other_eval = trim_after_first_meter(truth, other, axis_mode, freq_truth, freq_other)
    return compute_metrics(truth_eval, other_eval, freq_truth, freq_other)


def plot_metrics_box(title: str, stats_per_run: Dict[str, Dict[str, Dict[str, float]]]):
    if not stats_per_run:
        return
    metrics = METRIC_LABELS
    run_ids = sorted(stats_per_run.keys())
    models = sorted({model for stats in stats_per_run.values() for model in stats.keys()})
    if not models:
        return
    positions = []
    data = []
    tick_positions = []
    tick_labels = []
    group_width = 0.4
    base = 0.0
    for key, label in metrics:
        for j, model in enumerate(models):
            vals = [stats_per_run[r].get(model, {}).get(key, float("nan")) for r in run_ids]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                vals = [float("nan")]
            data.append(vals)
            positions.append(base + j * group_width)
        tick_positions.append(base + (len(models) - 1) * group_width / 2.0)
        tick_labels.append(label)
        base += len(models) * group_width + 0.6
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, positions=positions, widths=0.25, patch_artist=True, manage_ticks=False)
    palette = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for i, box in enumerate(bp["boxes"]):
        color = palette[i % len(models)]
        box.set_facecolor(color)
        box.set_alpha(0.7)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=palette[i], alpha=0.7) for i in range(len(models))],
        labels=models,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_diff_distributions(label: str, stats_per_run: Dict[str, Dict[str, Dict[str, float]]]):
    fig, axes = plt.subplots(1, len(METRIC_LABELS), figsize=(4 * len(METRIC_LABELS), 4))
    if len(METRIC_LABELS) == 1:
        axes = [axes]
    for ax, (key, metric_label) in zip(axes, METRIC_LABELS):
        diffs = []
        for run_stats in stats_per_run.values():
            cfd = run_stats.get("CFD", {}).get(key, float("nan"))
            cuboid = run_stats.get("Cuboid", {}).get(key, float("nan"))
            if np.isnan(cfd) or np.isnan(cuboid):
                continue
            diffs.append(cfd - cuboid)
        if diffs:
            ax.hist(diffs, bins=20, alpha=0.8, color="tab:blue")
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("d_i (CFD - Cuboid)")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(metric_label)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.suptitle(f"Differences d_i (CFD - Cuboid) - {label}")
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    plt.show()


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def paired_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    if std == 0:
        return float("inf"), 0.0
    t_stat = mean / (std / sqrt(n))
    try:
        from scipy import stats as scipy_stats  # type: ignore
        p_val = float(scipy_stats.t.sf(abs(t_stat), df=n - 1) * 2.0)
    except Exception:
        p_val = float(2.0 * (1.0 - normal_cdf(abs(t_stat))))
    return t_stat, p_val


def wilcoxon_signed_rank(a: List[float], b: List[float]) -> Tuple[float, float]:
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    diffs = np.array(a) - np.array(b)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0
    ranks = np.argsort(np.abs(diffs))
    rank_vals = np.empty_like(ranks, dtype=float)
    rank_vals[ranks] = np.arange(1, n + 1)
    w_pos = float(np.sum(rank_vals[diffs > 0]))
    w_neg = float(np.sum(rank_vals[diffs < 0]))
    stat = min(w_pos, w_neg)
    try:
        from scipy import stats as scipy_stats  # type: ignore
        p_val = float(scipy_stats.wilcoxon(a, b).pvalue)
    except Exception:
        mean = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        z = (stat - mean) / sqrt(var) if var > 0 else 0.0
        p_val = float(2.0 * (1.0 - normal_cdf(abs(z))))
    return stat, p_val


def discover_test_ids(base_dir: Path) -> List[int]:
    tests = []
    for path in base_dir.glob("Curee Test*.json"):
        match = re.search(r"Curee Test(\d+)", path.name)
        if match:
            tests.append(int(match.group(1)))
    return sorted(set(tests))


def discover_run_ids_from_bags(base_dir: Path, bag_prefix: str) -> List[int]:
    runs = []
    for path in base_dir.glob(f"{bag_prefix}*.bag"):
        match = re.search(rf"{bag_prefix}(\d+)", path.name)
        if match:
            runs.append(int(match.group(1)))
    return sorted(set(runs))


def build_run_mapping(base_dir: Path, bag_prefix: str, exclude_tests: set) -> Dict[int, int]:
    tests = discover_test_ids(base_dir)
    runs = discover_run_ids_from_bags(base_dir, bag_prefix)
    if len(runs) != len(tests):
        min_len = min(len(runs), len(tests))
        print(
            f"Warning: {base_dir} has {len(runs)} runs and {len(tests)} tests; using first {min_len}."
        )
    mapping = {run_id: test_id for run_id, test_id in zip(runs, tests)}
    mapping = {run_id: test_id for run_id, test_id in mapping.items() if test_id not in exclude_tests}
    return mapping


def pick_path(base_dir: Path, base: str, run_id: int, ext: str) -> Path:
    numbered = base_dir / f"{base}{run_id}{ext}"
    if numbered.exists():
        return numbered
    fallback = base_dir / f"{base}{ext}"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Neither {numbered} nor {fallback} found")


def collect_dataset_stats(
    dataset: Dict,
) -> Tuple[
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Dict[str, Dict[str, float]]],
]:
    base_dir = dataset["base_dir"]
    mapping = build_run_mapping(base_dir, dataset["bag_prefix"], dataset["exclude_tests"])
    qualy_starts = load_start_times(base_dir / dataset["start_times"], dataset["qualy_prefix"])
    mitre_starts = load_start_times(base_dir / dataset["start_times"], dataset["mitre_prefix"])
    raw_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    smooth_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    cut_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for run_id, qualy_id in mapping.items():
        try:
            gt_path = base_dir / f"Curee Test{qualy_id:04d}.json"
            odom_path = pick_path(base_dir, dataset["odom_prefix"], run_id, ".csv")
            cfd_path = pick_path(base_dir, dataset["cfd_prefix"], run_id, ".csv")
            cuboid_path = pick_path(base_dir, dataset["cuboid_prefix"], run_id, ".csv")
        except FileNotFoundError as exc:
            print(f"Skipping {dataset['name']} run {run_id}: {exc}")
            continue

        truth_full = load_ground_truth(gt_path)
        odom_full = load_xyz_csv(odom_path)
        cfd_full = load_xyz_csv(cfd_path)
        cuboid_full = load_xyz_csv(cuboid_path)

        gt_dur = ground_truth_duration_seconds(gt_path)
        odom_dur = len(odom_full) / 50.0 if len(odom_full) > 0 else 0.0
        cfd_dur = len(cfd_full) / 50.0 if len(cfd_full) > 0 else 0.0
        cuboid_dur = len(cuboid_full) / 50.0 if len(cuboid_full) > 0 else 0.0
        start_gt = qualy_starts.get(qualy_id, qualy_starts.get(run_id, 0.0))
        start_model = mitre_starts.get(run_id, 0.0)
        gt_avail = max(0.0, gt_dur - start_gt)
        odom_avail = max(0.0, odom_dur - start_model)
        cfd_avail = max(0.0, cfd_dur - start_model)
        cuboid_avail = max(0.0, cuboid_dur - start_model)
        plot_window = max(0.0, min(gt_avail, odom_avail, cfd_avail, cuboid_avail))

        truth = trim_by_time(truth_full, 100.0, start_gt)
        truth = limit_by_time(truth, 100.0, plot_window)
        odom = trim_by_time(odom_full, 50.0, start_model)
        odom = limit_by_time(odom, 50.0, plot_window)
        cfd = trim_by_time(cfd_full, 50.0, start_model)
        cfd = limit_by_time(cfd, 50.0, plot_window)
        cuboid = trim_by_time(cuboid_full, 50.0, start_model)
        cuboid = limit_by_time(cuboid, 50.0, plot_window)

        pre_trim_seconds = float(dataset.get("pre_trim_seconds", 0.0))
        if pre_trim_seconds > 0.0:
            truth = trim_by_samples(truth, int(round(pre_trim_seconds * 100.0)))
            odom = trim_by_samples(odom, int(round(pre_trim_seconds * 50.0)))
            cfd = trim_by_samples(cfd, int(round(pre_trim_seconds * 50.0)))
            cuboid = trim_by_samples(cuboid, int(round(pre_trim_seconds * 50.0)))

        run_key = f"{dataset['name']}-{run_id}"

        axis_mode = dataset.get("axis_mode", "xy")
        align_R_cfd = compute_alignment_rotation(truth, cfd, axis_mode)
        align_R_cuboid = compute_alignment_rotation(truth, cuboid, axis_mode)
        cfd_aligned_raw = apply_rotation_with_translation(cfd, align_R_cfd, truth[0])
        cuboid_aligned_raw = apply_rotation_with_translation(cuboid, align_R_cuboid, truth[0])
        raw_stats[run_key] = {
            "CFD": compute_metrics_after_meter(truth, cfd_aligned_raw, axis_mode, 100.0, 50.0),
            "Cuboid": compute_metrics_after_meter(truth, cuboid_aligned_raw, axis_mode, 100.0, 50.0),
        }

        truth_smooth = spike_smoothing(truth)
        cfd_aligned = apply_rotation_with_translation(cfd, align_R_cfd, truth_smooth[0])
        cuboid_aligned = apply_rotation_with_translation(cuboid, align_R_cuboid, truth_smooth[0])
        smooth_stats[run_key] = {
            "CFD": compute_metrics_after_meter(truth_smooth, cfd_aligned, axis_mode, 100.0, 50.0),
            "Cuboid": compute_metrics_after_meter(truth_smooth, cuboid_aligned, axis_mode, 100.0, 50.0),
        }

        truth_smooth_2s = trim_by_samples(truth_smooth, 200)
        cfd_2s = trim_by_samples(cfd, 100)
        cuboid_2s = trim_by_samples(cuboid, 100)
        cfd_2s_aligned = apply_rotation_with_translation(cfd_2s, align_R_cfd, truth_smooth_2s[0])
        cuboid_2s_aligned = apply_rotation_with_translation(cuboid_2s, align_R_cuboid, truth_smooth_2s[0])
        cut_stats[run_key] = {
            "CFD": compute_metrics_after_meter(truth_smooth_2s, cfd_2s_aligned, axis_mode, 100.0, 50.0),
            "Cuboid": compute_metrics_after_meter(truth_smooth_2s, cuboid_2s_aligned, axis_mode, 100.0, 50.0),
        }

    return raw_stats, smooth_stats, cut_stats


def _print_ttests(label: str, stats: Dict[str, Dict[str, Dict[str, float]]]):
    print(f"\nPaired t-tests (CFD vs Cuboid) - {label}")
    for key, metric_label in METRIC_LABELS:
        cfd_vals = []
        cuboid_vals = []
        for run_stats in stats.values():
            cfd = run_stats.get("CFD", {}).get(key, float("nan"))
            cuboid = run_stats.get("Cuboid", {}).get(key, float("nan"))
            if np.isnan(cfd) or np.isnan(cuboid):
                continue
            cfd_vals.append(cfd)
            cuboid_vals.append(cuboid)
        t_stat, p_val = paired_t_test(cfd_vals, cuboid_vals)
        w_stat, w_p = wilcoxon_signed_rank(cfd_vals, cuboid_vals)
        print(
            f"{metric_label}: n={len(cfd_vals)}, t={t_stat:.3f}, p={p_val:.3g}, "
            f"wilcoxon W={w_stat:.3f}, p={w_p:.3g}"
        )


def main():
    all_raw: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_smooth: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_cut: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in DATASETS:
        raw_stats, smooth_stats, cut_stats = collect_dataset_stats(dataset)
        all_raw.update(raw_stats)
        all_smooth.update(smooth_stats)
        all_cut.update(cut_stats)

    if not all_raw and not all_smooth and not all_cut:
        print("No runs found. Check dataset paths and file availability.")
        return

    plot_metrics_box("All runs: Error metrics (raw GT)", all_raw)
    plot_metrics_box("All runs: Error metrics (smoothed GT)", all_smooth)
    plot_metrics_box("All runs: Error metrics (smoothed GT after 2 s cut)", all_cut)
    plot_diff_distributions("raw GT", all_raw)
    plot_diff_distributions("smoothed GT", all_smooth)
    plot_diff_distributions("smoothed GT after 2 s cut", all_cut)

    _print_ttests("raw GT", all_raw)
    _print_ttests("smoothed GT", all_smooth)
    _print_ttests("smoothed GT after 2 s cut", all_cut)


if __name__ == "__main__":
    main()
