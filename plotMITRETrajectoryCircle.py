#!/usr/bin/env python3
"""
Plot MITRE Circle trajectories with yaw-only alignment.

Generated plots per run_id:
1) Ground truth vs odometry (aligned)
2) Ground truth with Cuboid, CFD, and Transient CFD (aligned)
3) Same as #2 after removing first 2 seconds
4) Ground truth (Kalman+RTS-smoothed) with Cuboid, CFD, and Transient CFD (aligned)
5) Same as #4 after removing first 2 seconds
6) Metric bar plots (raw GT + Kalman+RTS-smoothed GT)
7) Metric box plots across runs (raw GT + Kalman+RTS-smoothed GT)

Alignment notes:
- Use a yaw-only Umeyama fit on the first 1.5 meters of motion (after start trimming).
- Odom is aligned to GT using its first 1.5-meter segment; CFD and Cuboid are each aligned to GT separately.
- All aligned trajectories are translated to start at the GT origin.

Start times:
- startingCirclesTimes.json provides a start time (seconds) per run under keys
  qualyCircle{i} and mitreCircle{i}; these are used for trimming aligned trajectories.
- Ground truth sampling: 100 Hz.
- Odometry: 50 Hz.
- CFD/Cuboid/Transient CFD: 50 Hz (dt = 1/50).

Mapping notes:
- Run IDs (from bag filenames) are paired with Qualisys tests by sorted order.
- Tests {7, 12} are excluded before processing.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE_DIR = Path("MITRE/Circles")
START_TIMES_JSON = BASE_DIR / "startingCirclesTimes.json"
ALIGN_DISTANCE_M = 1.5
ALIGN_AXIS_MODE = "xy"
EXCLUDED_TEST_IDS = {7, 12}

BAG_PREFIX = "mitreCircle"
ODOM_PREFIX = "odomMITRECircle"
CFD_PREFIX = "positionsMITRECircleCFD"
CFD_TRANSIENT_PREFIX = "positionsMITRECircleTransientCFD"
CUBOID_PREFIX = "positionsMITRECircleCuboid"
QUALY_PREFIX = "qualyCircle"
MITRE_PREFIX = "mitreCircle"

PREFERRED_MARKER = "CUREE - 1"
FALLBACK_MARKER_PREFIX = "CUREE"


def discover_test_ids() -> List[int]:
    tests = []
    for path in BASE_DIR.glob("Curee Test*.json"):
        match = re.search(r"Curee Test(\d+)", path.name)
        if match:
            tests.append(int(match.group(1)))
    return sorted(set(tests))


def discover_run_ids_from_bags() -> List[int]:
    runs = []
    for path in BASE_DIR.glob(f"{BAG_PREFIX}*.bag"):
        match = re.search(rf"{BAG_PREFIX}(\d+)", path.name)
        if match:
            runs.append(int(match.group(1)))
    return sorted(set(runs))


def build_run_mapping() -> Dict[int, int]:
    tests = discover_test_ids()
    runs = discover_run_ids_from_bags()
    if len(runs) != len(tests):
        raise ValueError(
            f"Run/test count mismatch in {BASE_DIR}: {len(runs)} runs vs {len(tests)} tests."
        )
    mapping = {run_id: test_id for run_id, test_id in zip(runs, tests)}
    mapping = {
        run_id: test_id for run_id, test_id in mapping.items() if test_id not in EXCLUDED_TEST_IDS
    }
    if not mapping:
        raise ValueError(f"No valid runs remain after excluding {sorted(EXCLUDED_TEST_IDS)}.")
    return mapping


RUN_ID_TO_QUALY_ID = build_run_mapping()
AVAILABLE_RUN_IDS = tuple(sorted(RUN_ID_TO_QUALY_ID.keys()))


# ----------------------------
# Data loading helpers
# ----------------------------
def load_xyz_csv(path: Path) -> np.ndarray:
    """Load CSV with header x,y,z into (N,3) array."""
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _part_length(part: Dict) -> int:
    """Return sample count for a part using Range if available, else Values length."""
    range_info = part.get("Range") or {}
    start = range_info.get("Start")
    end = range_info.get("End")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return end - start + 1
    values = part.get("Values", [])
    return len(values) if isinstance(values, list) else 0


def _contiguous_lengths(parts: List[Dict]) -> Tuple[int, int]:
    """Return (max contiguous length, total length) inferred from parts."""
    best = 0
    total = 0
    for part in parts:
        length = _part_length(part)
        if length > best:
            best = length
        total += length
    return best, total


def _best_entry(entries: List[Dict], name_prefix: Optional[str] = None) -> Optional[Dict]:
    """Pick the entry with the longest contiguous length, optionally filtered by name prefix."""
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
    """Return xyz positions (m) for a marker or None if it has no values."""
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
    return np.vstack(chunks) / 1000.0  # mm -> m


def _rigid_body_sample_xyz(sample: object) -> Optional[List[float]]:
    if isinstance(sample, (list, tuple)) and sample:
        head = sample[0]
        if isinstance(head, (list, tuple)) and len(head) >= 3:
            return list(head[:3])
        if len(sample) >= 3 and all(isinstance(x, (int, float)) for x in sample[:3]):
            return list(sample[:3])
    return None


def _rigid_body_positions(rigid_body: Dict) -> Optional[np.ndarray]:
    """Return xyz positions (m) for a rigid body or None if it has no values."""
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
    """Choose preferred marker, then other CUREE, then longest marker; fallback to rigid body."""
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
    """Extract xyz (m) from Qualisys JSON using preferred source selection."""
    with path.open("r") as f:
        data = json.load(f)
    source_type, entry = _select_ground_truth_entry(data)
    name = entry.get("Name", "<unknown>")
    if source_type == "marker":
        positions = _marker_positions(entry)
    else:
        positions = _rigid_body_positions(entry)
    if positions is None:
        raise ValueError(f"Selected {source_type} '{name}' has no Values in Parts")
    return positions


def load_start_times(prefix: str = QUALY_PREFIX) -> Dict[int, float]:
    """Return start times keyed by run_id using entries like '<prefix>{i}' in JSON."""
    if not START_TIMES_JSON.exists():
        raise FileNotFoundError(f"Missing start times file: {START_TIMES_JSON}")
    with START_TIMES_JSON.open("r") as f:
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


def resolve_qualy_id(run_id: int) -> int:
    """Return the Qualisys test ID for a given run_id."""
    try:
        return RUN_ID_TO_QUALY_ID[run_id]
    except KeyError as exc:
        valid = ", ".join(str(r) for r in AVAILABLE_RUN_IDS)
        raise ValueError(f"Run {run_id} is not available (valid runs: {valid}).") from exc


def ground_truth_duration_seconds(run_id: int) -> float:
    """Compute GT duration from Qualisys JSON using the selected source."""
    qualy_id = resolve_qualy_id(run_id)
    gt_path = BASE_DIR / f"Curee Test{qualy_id:04d}.json"
    with gt_path.open("r") as f:
        data = json.load(f)
    _, entry = _select_ground_truth_entry(data)
    parts = entry.get("Parts", [])
    span = _parts_sample_span(parts)
    if span is not None:
        min_start, max_end = span
        return max(0.0, (max_end - min_start) / 100.0)
    _, total = _contiguous_lengths(parts)
    if total <= 0:
        return 0.0
    return max(0.0, (total - 1) / 100.0)


# ----------------------------
# Trimming helpers
# ----------------------------
def trim_by_samples(traj: np.ndarray, n: int) -> np.ndarray:
    """Trim the first n samples; keep at least one sample."""
    if n <= 0 or len(traj) == 0:
        return traj
    if n >= len(traj):
        return traj[-1:].copy()
    return traj[n:]


def trim_by_time(traj: np.ndarray, freq_hz: float, start_time: float) -> np.ndarray:
    """Remove all samples occurring before start_time (seconds)."""
    start_idx = int(start_time * freq_hz)
    return trim_by_samples(traj, start_idx)


def limit_by_time(traj: np.ndarray, freq_hz: float, max_seconds: float) -> np.ndarray:
    """Take at most max_seconds worth of samples from the start of traj based on sampling freq."""
    if len(traj) == 0:
        return traj
    if max_seconds <= 0:
        return traj[:1]
    n = int(max_seconds * freq_hz)
    if n < 2:
        n = 2
    return traj[: min(len(traj), n)]

# ----------------------------
# Alignment helpers (Umeyama)
# ----------------------------
def umeyama_align(
    source: np.ndarray,
    target: np.ndarray,
    fit_target: Optional[np.ndarray] = None,
    max_seconds: float = 6.0,
    freq_source: float = 100.0,
    freq_target: float = 100.0,
) -> np.ndarray:
    """
    Compute rotation+translation (scale=1) aligning source -> target using Umeyama,
    restricted to a yaw (x-y plane) rotation. Translation anchors starts together.
    Fit uses fit_target if provided; otherwise uses target. Uses up to max_seconds worth of data,
    with points determined by the sampling rates.
    """
    if len(source) == 0 or len(target) == 0:
        return source

    tgt_full = fit_target if fit_target is not None else target
    src_fit = limit_by_time(source, freq_source, max_seconds)
    tgt_fit = limit_by_time(tgt_full, freq_target, max_seconds)

    # Adjust for differing sampling rates: downsample the higher-rate sequence
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
    R2 = U @ S2 @ Vt  # 2x2 yaw rotation

    R = np.eye(3)
    R[:2, :2] = R2

    # Enforce initial condition
    t = target[0] - R @ source[0]

    aligned = (R @ source.T).T + t
    return aligned


# ----------------------------
# Smoothing helper (robust Kalman + RTS)
# ----------------------------
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
    state_dim = 2 * dim  # pos + vel

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


# ----------------------------
# Similarity metrics
# ----------------------------
def compute_velocity(traj: np.ndarray, freq_hz: float) -> np.ndarray:
    """Finite-difference velocity (same length minus 1)."""
    if len(traj) < 2:
        return np.empty((0, 3))
    return np.diff(traj, axis=0) * freq_hz


def resample_to_times(traj: np.ndarray, freq_hz: float, times: np.ndarray) -> np.ndarray:
    """Resample a trajectory onto a target time vector using per-axis interpolation."""
    if len(traj) == 0 or len(times) == 0:
        return np.empty((0, 3))
    t_in = np.arange(len(traj)) / freq_hz
    out = np.zeros((len(times), traj.shape[1]))
    for d in range(traj.shape[1]):
        out[:, d] = np.interp(times, t_in, traj[:, d])
    return out


def build_common_time_grid(len_a: int, len_b: int, freq_a: float, freq_b: float) -> Tuple[np.ndarray, float]:
    """Return a common time grid up to the shorter duration, sampled at min(freq_a, freq_b)."""
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
    """Discrete Fréchet distance between two polylines."""
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
    """Compute similarity metrics with time-aware resampling (uses min duration, sampled at min freq)."""
    times, freq_ref = build_common_time_grid(len(truth), len(other), freq_truth, freq_other)
    if times.size == 0:
        return {
            k: float("nan")
            for k in (
                "rmse_pos",
                "mae_pos",
                "rmse_xy",
                "mae_xy",
                "rmse_z",
                "mae_z",
                "vel_rmse",
                "vel_mae",
                "frechet",
            )
        }

    t_interp = resample_to_times(truth, freq_truth, times)
    o_interp = resample_to_times(other, freq_other, times)
    diff = o_interp - t_interp

    pos_err = np.linalg.norm(diff, axis=1)
    xy_err = np.linalg.norm(diff[:, :2], axis=1)
    z_err = np.abs(diff[:, 2])

    rmse_pos = float(np.sqrt(np.mean(pos_err ** 2)))
    mae_pos = float(np.mean(pos_err))
    rmse_xy = float(np.sqrt(np.mean(xy_err ** 2)))
    mae_xy = float(np.mean(xy_err))
    rmse_z = float(np.sqrt(np.mean(z_err ** 2)))
    mae_z = float(np.mean(z_err))

    tv = compute_velocity(t_interp, freq_ref)
    ov = compute_velocity(o_interp, freq_ref)
    m = min(len(tv), len(ov))
    if m > 0:
        vel_diff = ov[:m] - tv[:m]
        vel_err = np.linalg.norm(vel_diff, axis=1)
        vel_rmse = float(np.sqrt(np.mean(vel_err ** 2)))
        vel_mae = float(np.mean(vel_err))
    else:
        vel_rmse = float("nan")
        vel_mae = float("nan")

    frechet = discrete_frechet(t_interp, o_interp)
    return {
        "rmse_pos": rmse_pos,
        "mae_pos": mae_pos,
        "rmse_xy": rmse_xy,
        "mae_xy": mae_xy,
        "rmse_z": rmse_z,
        "mae_z": mae_z,
        "vel_rmse": vel_rmse,
        "vel_mae": vel_mae,
        "frechet": frechet,
    }


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


METRIC_LABELS = [
    ("rmse_pos", "RMSE position (m)"),
    ("mae_pos", "MAE position (m)"),
    ("rmse_xy", "RMSE horizontal (m)"),
    ("mae_xy", "MAE horizontal (m)"),
    ("rmse_z", "RMSE vertical (m)"),
    ("mae_z", "MAE vertical (m)"),
    ("vel_rmse", "Velocity RMSE (m/s)"),
    ("vel_mae", "Velocity MAE (m/s)"),
    ("frechet", "Frechet distance (m)"),
]


def plot_metrics_bar(title: str, stats_by_model: Dict[str, Dict[str, float]]):
    """Bar plot of metrics for CFD/Cuboid vs ground truth."""
    metrics = METRIC_LABELS
    models = list(stats_by_model.keys())
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [stats_by_model[model].get(key, float("nan")) for key, _ in metrics]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_metrics_box(title: str, stats_per_run: Dict[int, Dict[str, Dict[str, float]]]):
    """Box plots of metrics across runs (keys: run_id -> model -> metrics)."""
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
            vals = [
                stats_per_run[r].get(model, {}).get(key, float("nan"))
                for r in run_ids
            ]
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
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=palette[i], alpha=0.7) for i in range(len(models))], labels=models)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Plotting helpers
# ----------------------------
def set_equal_axes(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    max_range = (maxs - mins).max()
    mids = 0.5 * (maxs + mins)
    ax.set_xlim(mids[0] - max_range / 2, mids[0] + max_range / 2)
    ax.set_ylim(mids[1] - max_range / 2, mids[1] + max_range / 2)
    ax.set_zlim(mids[2] - max_range / 2, mids[2] + max_range / 2)


def plot_two(title: str, truth: np.ndarray, other: np.ndarray, labels: Tuple[str, str]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], label=labels[0], linewidth=2)
    ax.plot(other[:, 0], other[:, 1], other[:, 2], label=labels[1], linewidth=2)
    ax.scatter(truth[0, 0], truth[0, 1], truth[0, 2], s=50, marker="o", label="start (GT)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    all_pts = np.vstack([truth, other])
    set_equal_axes(ax, all_pts)
    plt.tight_layout()
    plt.show()


def plot_three(title: str, truth: np.ndarray, cfd: np.ndarray, cuboid: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], label="ground truth", linewidth=2)
    ax.plot(cfd[:, 0], cfd[:, 1], cfd[:, 2], label="CFD", linewidth=2, linestyle=":")
    ax.plot(cuboid[:, 0], cuboid[:, 1], cuboid[:, 2], label="Cuboid", linewidth=2, linestyle="--")
    ax.scatter(truth[0, 0], truth[0, 1], truth[0, 2], s=50, marker="o", label="start (GT)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    all_pts = np.vstack([truth, cfd, cuboid])
    set_equal_axes(ax, all_pts)
    plt.tight_layout()
    plt.show()


def plot_four(title: str, truth: np.ndarray, cfd: np.ndarray, cfd_transient: np.ndarray, cuboid: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], label="ground truth", linewidth=2)
    ax.plot(cfd[:, 0], cfd[:, 1], cfd[:, 2], label="CFD", linewidth=2, linestyle=":")
    ax.plot(
        cfd_transient[:, 0],
        cfd_transient[:, 1],
        cfd_transient[:, 2],
        label="Transient CFD",
        linewidth=2,
        linestyle="-.",
    )
    ax.plot(cuboid[:, 0], cuboid[:, 1], cuboid[:, 2], label="Cuboid", linewidth=2, linestyle="--")
    ax.scatter(truth[0, 0], truth[0, 1], truth[0, 2], s=50, marker="o", label="start (GT)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    all_pts = np.vstack([truth, cfd, cfd_transient, cuboid])
    set_equal_axes(ax, all_pts)
    plt.tight_layout()
    plt.show()


def plot_one(title: str, truth: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], label="ground truth", linewidth=2)
    ax.scatter(truth[0, 0], truth[0, 1], truth[0, 2], s=50, marker="o", label="start (GT)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    set_equal_axes(ax, truth)
    plt.tight_layout()
    plt.show()


# ----------------------------
# File selection
# ----------------------------
def pick_path(base: str, run_id: int, ext: str) -> Path:
    """Choose numbered file if present, else fallback to unnumbered."""
    numbered = BASE_DIR / f"{base}{run_id}{ext}"
    if numbered.exists():
        return numbered
    fallback = BASE_DIR / f"{base}{ext}"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Neither {numbered} nor {fallback} found")


def pick_cfd_paths(run_id: int) -> Tuple[Path, Path]:
    """Return standard and transient CFD paths for a run."""
    cfd_path = pick_path(CFD_PREFIX, run_id, ".csv")
    transient_path = pick_path(CFD_TRANSIENT_PREFIX, run_id, ".csv")
    return cfd_path, transient_path


def load_and_align_run(run_id: int, qualy_starts=None, mitre_starts=None) -> Dict[str, np.ndarray]:
    """Load data for a run, trim by overlapping time window, and produce aligned trajectories."""
    qualy_id = resolve_qualy_id(run_id)
    if qualy_starts is None:
        qualy_starts = load_start_times(prefix=QUALY_PREFIX)
    if mitre_starts is None:
        mitre_starts = load_start_times(prefix=MITRE_PREFIX)
    start_gt = qualy_starts.get(qualy_id, qualy_starts.get(run_id, 0.0))
    start_model = mitre_starts.get(run_id, 0.0)

    gt_path = BASE_DIR / f"Curee Test{qualy_id:04d}.json"
    odom_path = pick_path(ODOM_PREFIX, run_id, ".csv")
    cfd_path, cfd_transient_path = pick_cfd_paths(run_id)
    cuboid_path = pick_path(CUBOID_PREFIX, run_id, ".csv")

    truth_full = load_ground_truth(gt_path)
    odom_full = load_xyz_csv(odom_path)
    cfd_full = load_xyz_csv(cfd_path)
    cfd_transient_full = load_xyz_csv(cfd_transient_path)
    cuboid_full = load_xyz_csv(cuboid_path)

    gt_dur = ground_truth_duration_seconds(run_id)
    odom_dur = len(odom_full) / 50.0 if len(odom_full) > 0 else 0.0
    cfd_dur = len(cfd_full) / 50.0 if len(cfd_full) > 0 else 0.0
    cfd_transient_dur = len(cfd_transient_full) / 50.0 if len(cfd_transient_full) > 0 else 0.0
    cuboid_dur = len(cuboid_full) / 50.0 if len(cuboid_full) > 0 else 0.0
    gt_avail = max(0.0, gt_dur - start_gt)
    odom_avail = max(0.0, odom_dur - start_model)
    cfd_avail = max(0.0, cfd_dur - start_model)
    cfd_transient_avail = max(0.0, cfd_transient_dur - start_model)
    cuboid_avail = max(0.0, cuboid_dur - start_model)
    plot_window = max(0.0, min(gt_avail, odom_avail, cfd_avail, cfd_transient_avail, cuboid_avail))

    truth = trim_by_time(truth_full, 100.0, start_gt)
    truth = limit_by_time(truth, 100.0, plot_window)
    odom = trim_by_time(odom_full, 50.0, start_model)
    odom = limit_by_time(odom, 50.0, plot_window)
    cfd = trim_by_time(cfd_full, 50.0, start_model)
    cfd = limit_by_time(cfd, 50.0, plot_window)
    cfd_transient = trim_by_time(cfd_transient_full, 50.0, start_model)
    cfd_transient = limit_by_time(cfd_transient, 50.0, plot_window)
    cuboid = trim_by_time(cuboid_full, 50.0, start_model)
    cuboid = limit_by_time(cuboid, 50.0, plot_window)

    truth_smooth = spike_smoothing(truth)

    align_R_odom = compute_alignment_rotation(truth, odom, ALIGN_AXIS_MODE)
    align_R_cfd = compute_alignment_rotation(truth, cfd, ALIGN_AXIS_MODE)
    align_R_cfd_transient = compute_alignment_rotation(truth, cfd_transient, ALIGN_AXIS_MODE)
    align_R_cuboid = compute_alignment_rotation(truth, cuboid, ALIGN_AXIS_MODE)

    odom_aligned = apply_rotation_with_translation(odom, align_R_odom, truth[0])
    cfd_aligned = apply_rotation_with_translation(cfd, align_R_cfd, truth[0])
    cfd_transient_aligned = apply_rotation_with_translation(cfd_transient, align_R_cfd_transient, truth[0])
    cuboid_aligned = apply_rotation_with_translation(cuboid, align_R_cuboid, truth[0])

    cfd_smooth = apply_rotation_with_translation(cfd, align_R_cfd, truth_smooth[0])
    cfd_transient_smooth = apply_rotation_with_translation(cfd_transient, align_R_cfd_transient, truth_smooth[0])
    cuboid_smooth = apply_rotation_with_translation(cuboid, align_R_cuboid, truth_smooth[0])

    return {
        "truth_full": truth_full,
        "truth": truth,
        "truth_smooth": truth_smooth,
        "odom": odom,
        "odom_aligned": odom_aligned,
        "cfd": cfd,
        "cfd_transient": cfd_transient,
        "cuboid": cuboid,
        "cfd_aligned": cfd_aligned,
        "cfd_transient_aligned": cfd_transient_aligned,
        "cuboid_aligned": cuboid_aligned,
        "cfd_smooth": cfd_smooth,
        "cfd_transient_smooth": cfd_transient_smooth,
        "cuboid_smooth": cuboid_smooth,
        "align_R_odom": align_R_odom,
        "align_R_cfd": align_R_cfd,
        "align_R_cfd_transient": align_R_cfd_transient,
        "align_R_cuboid": align_R_cuboid,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot MITRE Circle trajectories with yaw-only alignment.")
    parser.add_argument(
        "--run-id",
        type=int,
        default=AVAILABLE_RUN_IDS[0],
        choices=AVAILABLE_RUN_IDS,
        help=f"Episode number (valid runs: {', '.join(str(r) for r in AVAILABLE_RUN_IDS)})",
    )
    args = parser.parse_args()
    run_id = args.run_id
    qualy_starts = load_start_times(prefix=QUALY_PREFIX)
    mitre_starts = load_start_times(prefix=MITRE_PREFIX)

    run_data = load_and_align_run(run_id, qualy_starts, mitre_starts)
    truth_full = run_data["truth_full"]
    truth = run_data["truth"]
    truth_smooth = run_data["truth_smooth"]
    odom = run_data["odom"]
    odom_aligned = run_data["odom_aligned"]
    cfd = run_data["cfd"]
    cfd_transient = run_data["cfd_transient"]
    cuboid = run_data["cuboid"]
    cfd_aligned = run_data["cfd_aligned"]
    cfd_transient_aligned = run_data["cfd_transient_aligned"]
    cuboid_aligned = run_data["cuboid_aligned"]
    cfd_smooth = run_data["cfd_smooth"]
    cfd_transient_smooth = run_data["cfd_transient_smooth"]
    cuboid_smooth = run_data["cuboid_smooth"]
    align_R_cfd = run_data["align_R_cfd"]
    align_R_cfd_transient = run_data["align_R_cfd_transient"]
    align_R_cuboid = run_data["align_R_cuboid"]

    # Plot 0: Raw GT only
    plot_one(f"Run {run_id}: Ground Truth (raw, full)", truth_full)

    # Plot 1: GT vs Odom
    plot_two(f"Run {run_id}: Ground Truth vs Odom", truth, odom_aligned, labels=("ground truth", "odom"))

    # Plot 2: GT with CFD + Transient CFD + Cuboid
    plot_four(
        f"Run {run_id}: Ground Truth with CFD, Transient CFD, and Cuboid",
        truth,
        cfd_aligned,
        cfd_transient_aligned,
        cuboid_aligned,
    )

    # Plot 3: after removing first 2 seconds
    truth_2s = trim_by_samples(truth, 200)  # 2 s @ 100 Hz
    cfd_2s = trim_by_samples(cfd, 100)      # 2 s @ 50 Hz
    cfd_transient_2s = trim_by_samples(cfd_transient, 100)
    cuboid_2s = trim_by_samples(cuboid, 100)
    cfd_2s_aligned = apply_rotation_with_translation(cfd_2s, align_R_cfd, truth_2s[0])
    cfd_transient_2s_aligned = apply_rotation_with_translation(
        cfd_transient_2s, align_R_cfd_transient, truth_2s[0]
    )
    cuboid_2s_aligned = apply_rotation_with_translation(cuboid_2s, align_R_cuboid, truth_2s[0])
    plot_four(
        f"Run {run_id}: GT with CFD, Transient CFD, and Cuboid (after 2 s cut)",
        truth_2s,
        cfd_2s_aligned,
        cfd_transient_2s_aligned,
        cuboid_2s_aligned,
    )

    # Plot 4: Spike-smoothed GT with CFD + Transient CFD + Cuboid
    cfd_smooth = apply_rotation_with_translation(cfd, align_R_cfd, truth_smooth[0])
    cfd_transient_smooth = apply_rotation_with_translation(
        cfd_transient, align_R_cfd_transient, truth_smooth[0]
    )
    cuboid_smooth = apply_rotation_with_translation(cuboid, align_R_cuboid, truth_smooth[0])
    plot_four(
        f"Run {run_id}: Smoothed GT (Kalman+RTS-smoothed) with CFD, Transient CFD, and Cuboid",
        truth_smooth,
        cfd_smooth,
        cfd_transient_smooth,
        cuboid_smooth,
    )

    # Error metrics (raw GT vs aligned CFD/Transient CFD/Cuboid)
    raw_stats = {
        "CFD": compute_metrics_after_meter(truth, cfd_aligned, ALIGN_AXIS_MODE, 100.0, 50.0),
        "TransientCFD": compute_metrics_after_meter(truth, cfd_transient_aligned, ALIGN_AXIS_MODE, 100.0, 50.0),
        "Cuboid": compute_metrics_after_meter(truth, cuboid_aligned, ALIGN_AXIS_MODE, 100.0, 50.0),
    }

    # Plot 5: Spike-smoothed GT after 2 s cut
    truth_smooth_2s = trim_by_samples(truth_smooth, 200)
    cfd_smooth_2s = trim_by_samples(cfd, 100)
    cfd_transient_smooth_2s = trim_by_samples(cfd_transient, 100)
    cuboid_smooth_2s = trim_by_samples(cuboid, 100)
    cfd_smooth_2s_aligned = apply_rotation_with_translation(cfd_smooth_2s, align_R_cfd, truth_smooth_2s[0])
    cfd_transient_smooth_2s_aligned = apply_rotation_with_translation(
        cfd_transient_smooth_2s, align_R_cfd_transient, truth_smooth_2s[0]
    )
    cuboid_smooth_2s_aligned = apply_rotation_with_translation(cuboid_smooth_2s, align_R_cuboid, truth_smooth_2s[0])
    plot_four(
        f"Run {run_id}: Smoothed GT (Kalman+RTS-smoothed) with CFD, Transient CFD, and Cuboid after 2 s cut",
        truth_smooth_2s,
        cfd_smooth_2s_aligned,
        cfd_transient_smooth_2s_aligned,
        cuboid_smooth_2s_aligned,
    )

    # Error metrics (smoothed GT vs aligned CFD/Transient CFD/Cuboid)
    smooth_stats = {
        "CFD": compute_metrics_after_meter(truth_smooth, cfd_smooth, ALIGN_AXIS_MODE, 100.0, 50.0),
        "TransientCFD": compute_metrics_after_meter(truth_smooth, cfd_transient_smooth, ALIGN_AXIS_MODE, 100.0, 50.0),
        "Cuboid": compute_metrics_after_meter(truth_smooth, cuboid_smooth, ALIGN_AXIS_MODE, 100.0, 50.0),
    }
    cut_stats = {
        "CFD": compute_metrics_after_meter(truth_smooth_2s, cfd_smooth_2s_aligned, ALIGN_AXIS_MODE, 100.0, 50.0),
        "TransientCFD": compute_metrics_after_meter(truth_smooth_2s, cfd_transient_smooth_2s_aligned, ALIGN_AXIS_MODE, 100.0, 50.0),
        "Cuboid": compute_metrics_after_meter(truth_smooth_2s, cuboid_smooth_2s_aligned, ALIGN_AXIS_MODE, 100.0, 50.0),
    }

    # Final bar plots
    plot_metrics_bar(f"Run {run_id}: Error metrics (raw GT)", raw_stats)
    plot_metrics_bar(f"Run {run_id}: Error metrics (Kalman+RTS-smoothed GT)", smooth_stats)

    # Aggregate metrics across runs (1-5) for box plots
    all_raw_stats: Dict[int, Dict[str, Dict[str, float]]] = {run_id: raw_stats}
    all_smooth_stats: Dict[int, Dict[str, Dict[str, float]]] = {run_id: smooth_stats}
    all_cut_stats: Dict[int, Dict[str, Dict[str, float]]] = {run_id: cut_stats}

    for rid in AVAILABLE_RUN_IDS:
        if rid == run_id:
            continue
        rd = load_and_align_run(rid, qualy_starts, mitre_starts)
        all_raw_stats[rid] = {
            "CFD": compute_metrics_after_meter(rd["truth"], rd["cfd_aligned"], ALIGN_AXIS_MODE, 100.0, 50.0),
            "TransientCFD": compute_metrics_after_meter(rd["truth"], rd["cfd_transient_aligned"], ALIGN_AXIS_MODE, 100.0, 50.0),
            "Cuboid": compute_metrics_after_meter(rd["truth"], rd["cuboid_aligned"], ALIGN_AXIS_MODE, 100.0, 50.0),
        }
        all_smooth_stats[rid] = {
            "CFD": compute_metrics_after_meter(rd["truth_smooth"], rd["cfd_smooth"], ALIGN_AXIS_MODE, 100.0, 50.0),
            "TransientCFD": compute_metrics_after_meter(rd["truth_smooth"], rd["cfd_transient_smooth"], ALIGN_AXIS_MODE, 100.0, 50.0),
            "Cuboid": compute_metrics_after_meter(rd["truth_smooth"], rd["cuboid_smooth"], ALIGN_AXIS_MODE, 100.0, 50.0),
        }
        truth_smooth_2s_r = trim_by_samples(rd["truth_smooth"], 200)
        cfd_smooth_2s_r = trim_by_samples(rd["cfd"], 100)
        cfd_transient_smooth_2s_r = trim_by_samples(rd["cfd_transient"], 100)
        cuboid_smooth_2s_r = trim_by_samples(rd["cuboid"], 100)
        align_R_cfd_r = rd["align_R_cfd"]
        align_R_cfd_transient_r = rd["align_R_cfd_transient"]
        align_R_cuboid_r = rd["align_R_cuboid"]
        cfd_smooth_2s_aligned_r = apply_rotation_with_translation(cfd_smooth_2s_r, align_R_cfd_r, truth_smooth_2s_r[0])
        cfd_transient_smooth_2s_aligned_r = apply_rotation_with_translation(
            cfd_transient_smooth_2s_r, align_R_cfd_transient_r, truth_smooth_2s_r[0]
        )
        cuboid_smooth_2s_aligned_r = apply_rotation_with_translation(cuboid_smooth_2s_r, align_R_cuboid_r, truth_smooth_2s_r[0])
        all_cut_stats[rid] = {
            "CFD": compute_metrics_after_meter(truth_smooth_2s_r, cfd_smooth_2s_aligned_r, ALIGN_AXIS_MODE, 100.0, 50.0),
            "TransientCFD": compute_metrics_after_meter(
                truth_smooth_2s_r, cfd_transient_smooth_2s_aligned_r, ALIGN_AXIS_MODE, 100.0, 50.0
            ),
            "Cuboid": compute_metrics_after_meter(truth_smooth_2s_r, cuboid_smooth_2s_aligned_r, ALIGN_AXIS_MODE, 100.0, 50.0),
        }

    run_label = ", ".join(str(r) for r in AVAILABLE_RUN_IDS)
    plot_metrics_box(f"Runs {run_label}: Error metrics (raw GT)", all_raw_stats)
    plot_metrics_box(f"Runs {run_label}: Error metrics (Kalman+RTS-smoothed GT)", all_smooth_stats)
    plot_metrics_box(f"Runs {run_label}: Error metrics (Kalman+RTS-smoothed GT after 2 s cut)", all_cut_stats)


if __name__ == "__main__":
    main()
