#!/usr/bin/env python3
"""
Plot MITRE Left/Up body-frame velocity alongside left thruster motor rad/s per test.

For each Curee Test JSON in MITRE/Left or MITRE/Up, generate a 1x2 subplot figure.
Each subplot overlays x/y/z body-frame velocity, overall speed magnitude, and one thruster's motor rad/s.
Highlights are the constant-velocity windows that follow each PWM event.
Also generates a trajectory plot with body-frame velocity vectors over highlighted spans.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rosbag
from genpy.dynamic import generate_dynamic
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL_3D = True
except Exception:
    HAS_MPL_3D = False

LEFT_DIR = Path("MITRE/Left")
UP_DIR = Path("MITRE/Up")
BASE_DIR = UP_DIR
START_TIMES_JSON = UP_DIR / "startingUpTimes.json"
BAG_PREFIX = "mitreUp"
QUALY_PREFIX = "qualyUp"
MITRE_PREFIX = "mitreUp"
MODE_LABEL = "Up"

# Match exclusions used by plotMITRETrajectoryLeft/Up.
LEFT_EXCLUDED_TEST_IDS = {7}
UP_EXCLUDED_TEST_IDS = {7, 12}

PREFERRED_MARKER = "CUREE - 1"
FALLBACK_MARKER_PREFIX = "CUREE"

PWM_TOPIC = "/warpauv_1/control/motor_controller_feather/pwm_command_list"

DEADBAND = 0.08
FORWARD_SCALE = 0.8
FORWARD_A = -139.0
FORWARD_B = 500.0
FORWARD_C = 8.28
BACK_A = 161.0
BACK_B = 517.86
BACK_C = -5.72
FORWARD_BIAS = 1.0
BACK_BIAS = 0.5

DEFAULT_ROTOR_CONSTANT = 0.0002

# Keep the same order used in plotMITREPWM.py for consistent labeling.
THRUSTER_ORDER = (
    "drive_left",
    "drive_right",
    "rear_right",
    "front_right",
    "front_left",
    "rear_left",
)

PLOT_THRUSTERS = ("rear_left", "front_left")

DEFAULT_MIN_SPEED = 0.03
DEFAULT_MAX_SPEED_SLOPE = 0.080
DEFAULT_MIN_SEGMENT_SEC = 0.1
DEFAULT_MIN_VEL_SPIKE_SEC = 0.15
DEFAULT_MIN_AVG_ACCEL = 0.05
DEFAULT_MIN_MOTOR = 1e-3
DEFAULT_MAX_MOTOR_SLOPE = 5.0
DEFAULT_FORCE_CONST_RTOL = 1e-3
DEFAULT_FORCE_CONST_ATOL = 1e-3
DEFAULT_HIGHLIGHT_MIN_START = 3.0
DEFAULT_VEL_SPIKE_MAX = 0.5
DEFAULT_SPIKE_DERIV_FACTOR = 3.0
DEFAULT_SPIKE_DERIV_MIN = 0.1
DEFAULT_SPIKE_EXPAND_PAD = 1

DEFAULT_WATER_RHO = 997.0
DEFAULT_MASS = 25.219
DEFAULT_INERTIA = (0.30, 1.2, 1.07)

DEFAULT_DRAG_LOG_CUBOID_PREFIX = "drag_forces_moments_Cuboid"
DEFAULT_DRAG_LOG_CFD_PREFIX = "drag_forces_moments_CFD"

VECTOR_COLORS = {
    "rear_left": "tab:purple",
    "front_left": "tab:brown",
}

DEFAULT_DIR_EPS = 1e-6


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


def ground_truth_duration_seconds(path: Path) -> float:
    """Compute GT duration from Qualisys JSON using the selected source."""
    with path.open("r") as f:
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


def mad(arr: np.ndarray) -> float:
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def sg_velocity(
    traj: np.ndarray,
    freq_hz: float,
    window_length: int = 21,
    polyorder: int = 3,
) -> np.ndarray:
    if traj.shape[0] < 3:
        return np.empty((0, 3), dtype=float)
    n = traj.shape[0]
    win = min(window_length, n if n % 2 == 1 else n - 1)
    if win < 5:
        win = 5 if n >= 5 else (n if n % 2 == 1 else n - 1)
    if win < 3:
        return np.empty((0, 3), dtype=float)
    if polyorder >= win:
        polyorder = max(1, win - 2)
    dt = 1.0 / float(freq_hz)
    vel = savgol_filter(traj, window_length=win, polyorder=polyorder, deriv=1, delta=dt, axis=0, mode="interp")
    return vel[:-1]


def despike_velocity(
    vel: np.ndarray,
    freq_hz: float,
    window: int = 13,
) -> np.ndarray:
    if vel.size == 0:
        return vel
    win = window if window % 2 == 1 else window + 1
    win = max(3, min(win, vel.shape[0] if vel.shape[0] % 2 == 1 else vel.shape[0] - 1))
    if win < 3:
        return vel
    smoothed = np.empty_like(vel)
    for i in range(vel.shape[1]):
        smoothed[:, i] = savgol_filter(vel[:, i], window_length=win, polyorder=2, mode="interp")
    dt = 1.0 / float(freq_hz)
    spike_mask = np.any(np.abs(vel) > DEFAULT_VEL_SPIKE_MAX, axis=1)
    if not np.any(spike_mask):
        return vel

    dvel = np.gradient(vel, dt, axis=0)
    dmag = np.linalg.norm(dvel, axis=1)
    deriv_thresh = max(np.median(dmag) * DEFAULT_SPIKE_DERIV_FACTOR, DEFAULT_SPIKE_DERIV_MIN)

    idx = np.where(spike_mask)[0]
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = i
        prev = i
    runs.append((start, prev))

    expanded = np.zeros_like(spike_mask, dtype=bool)
    n = len(spike_mask)
    for start, end in runs:
        left = start
        while left > 0 and dmag[left] > deriv_thresh:
            left -= 1
        right = end
        while right < n - 1 and dmag[right] > deriv_thresh:
            right += 1
        left = max(0, left - DEFAULT_SPIKE_EXPAND_PAD)
        right = min(n - 1, right + DEFAULT_SPIKE_EXPAND_PAD)
        expanded[left : right + 1] = True

    cleaned = vel.copy()
    times = np.arange(vel.shape[0]) * dt
    for i in range(vel.shape[1]):
        series = cleaned[:, i].copy()
        series[expanded] = np.nan
        valid = ~np.isnan(series)
        if np.count_nonzero(valid) < 2:
            cleaned[:, i] = smoothed[:, i]
            continue
        cleaned[:, i] = np.interp(times, times[valid], series[valid])
    return cleaned


def trim_by_time(traj: np.ndarray, freq_hz: float, start_time: float) -> np.ndarray:
    if start_time <= 0 or len(traj) == 0:
        return traj
    start_idx = int(start_time * freq_hz)
    if start_idx >= len(traj):
        return traj[-1:].copy()
    return traj[start_idx:]


def limit_by_time(traj: np.ndarray, freq_hz: float, max_seconds: float) -> np.ndarray:
    if len(traj) == 0:
        return traj
    if max_seconds <= 0:
        return traj[:1]
    n = int(max_seconds * freq_hz)
    if n < 2:
        n = 2
    return traj[: min(len(traj), n)]


def limit_time_series(
    times: np.ndarray, values: np.ndarray, max_seconds: float
) -> Tuple[np.ndarray, np.ndarray]:
    if len(times) == 0 or max_seconds <= 0:
        return times[:0], values[:0]
    mask = times <= max_seconds
    if not np.any(mask):
        return times[:0], values[:0]
    return times[mask], values[mask]


def pwm_duration_seconds(times: np.ndarray) -> float:
    if len(times) == 0:
        return 0.0
    return max(0.0, float(times[-1]))


def interpolate_series(
    times_src: np.ndarray, values_src: np.ndarray, times_tgt: np.ndarray
) -> np.ndarray:
    if len(times_src) == 0:
        return np.full_like(times_tgt, np.nan, dtype=float)
    if len(times_src) == 1:
        return np.full_like(times_tgt, float(values_src[0]), dtype=float)
    out = np.full_like(times_tgt, np.nan, dtype=float)
    mask = (times_tgt >= times_src[0]) & (times_tgt <= times_src[-1])
    if np.any(mask):
        out[mask] = np.interp(times_tgt[mask], times_src, values_src)
    return out


def _normalize_vector(vec: np.ndarray, eps: float = DEFAULT_DIR_EPS) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < eps:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def overall_direction(positions: np.ndarray) -> np.ndarray:
    if positions.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    vec = positions[-1] - positions[0]
    if np.linalg.norm(vec) < DEFAULT_DIR_EPS:
        diffs = np.diff(positions, axis=0)
        vec = np.nanmean(diffs, axis=0)
    return _normalize_vector(vec)


def primary_direction_for_mode(mode_label: str) -> np.ndarray:
    if mode_label.lower() == "left":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    if mode_label.lower() == "up":
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return np.array([1.0, 0.0, 0.0], dtype=float)


def primary_velocity_component(vel_b: np.ndarray, mode_label: str) -> np.ndarray:
    if vel_b.size == 0:
        return vel_b
    if mode_label.lower() == "left":
        return vel_b[:, 1]
    if mode_label.lower() == "up":
        return vel_b[:, 2]
    return vel_b[:, 0]


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_apply_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm == 0.0:
        return v
    qn = q / norm
    qw = qn[0]
    qv = qn[1:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize_vector(a)
    b = _normalize_vector(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < -0.999999:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=float)
    axis = np.cross(a, b)
    q = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=float)
    norm = np.linalg.norm(q)
    if norm < DEFAULT_DIR_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


THRUSTER_RPY = {
    "rear_left": (0.0, -0.785398, 1.5708),
    "rear_right": (0.0, -0.785398, -1.5708),
    "front_left": (0.0, 0.785398, 1.5708),
    "front_right": (0.0, 0.785398, -1.5708),
    "drive_left": (0.0, 0.0, 0.0),
    "drive_right": (0.0, 0.0, 0.0),
}


def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix for XYZ (roll, pitch, yaw)."""
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def thruster_axes_body() -> Dict[str, np.ndarray]:
    axes: Dict[str, np.ndarray] = {}
    for name, (roll, pitch, yaw) in THRUSTER_RPY.items():
        axis = rot_from_rpy(roll, pitch, yaw).T @ np.array([1.0, 0.0, 0.0], dtype=float)
        axes[name] = axis
    return axes


def rotate_world_to_body(vectors_w: np.ndarray, q_wb: np.ndarray) -> np.ndarray:
    q_bw = quat_conjugate_np(q_wb)
    if vectors_w.ndim == 1:
        return quat_apply_np(q_bw, vectors_w)
    return np.array([quat_apply_np(q_bw, v) for v in vectors_w], dtype=float)


def constant_mask(
    times: np.ndarray,
    signal: np.ndarray,
    min_value: float,
    max_slope: float,
    use_abs: bool = False,
) -> np.ndarray:
    if len(times) < 2 or len(signal) != len(times):
        return np.zeros_like(times, dtype=bool)
    ds_dt = np.gradient(signal, times)
    mask = np.isfinite(signal) & np.isfinite(ds_dt)
    amp = np.abs(signal) if use_abs else signal
    if min_value > 0.0:
        mask &= np.abs(amp) >= min_value
    if max_slope > 0.0:
        mask &= np.abs(ds_dt) <= max_slope
    return mask


def segments_from_mask(
    times: np.ndarray,
    mask: np.ndarray,
    min_duration: float,
) -> List[Tuple[float, float]]:
    if len(times) == 0 or len(mask) == 0 or not np.any(mask):
        return []
    segments: List[Tuple[float, float]] = []
    start_idx = None
    for idx, ok in enumerate(mask):
        if ok and start_idx is None:
            start_idx = idx
        elif not ok and start_idx is not None:
            end_idx = idx - 1
            if (times[end_idx] - times[start_idx]) >= min_duration:
                segments.append((times[start_idx], times[end_idx]))
            start_idx = None
    if start_idx is not None:
        end_idx = len(times) - 1
        if (times[end_idx] - times[start_idx]) >= min_duration:
            segments.append((times[start_idx], times[end_idx]))
    return segments


def clip_segments(segments: List[Tuple[float, float]], min_start: float) -> List[Tuple[float, float]]:
    if min_start <= 0.0:
        return segments
    clipped: List[Tuple[float, float]] = []
    for seg_start, seg_end in segments:
        if seg_end <= min_start:
            continue
        clipped.append((max(seg_start, min_start), seg_end))
    return clipped


def _segments_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def find_overlapping_segment(
    seg: Tuple[float, float],
    candidates: List[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    for cand in candidates:
        if _segments_overlap(seg, cand):
            return cand
    return None


def compute_velocity_frames(
    positions_trim: np.ndarray,
    freq_hz: float,
    q_wb: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    velocity_raw = sg_velocity(positions_trim, freq_hz)
    if velocity_raw.size == 0:
        return None
    vel_b_raw = rotate_world_to_body(velocity_raw, q_wb)
    velocity_plot = despike_velocity(velocity_raw, freq_hz)
    vel_b_plot = rotate_world_to_body(velocity_plot, q_wb)
    speed = np.linalg.norm(vel_b_plot, axis=1)
    vx_raw = vel_b_raw[:, 0]
    return {
        "velocity_raw": velocity_raw,
        "vel_b_raw": vel_b_raw,
        "vel_b_plot": vel_b_plot,
        "speed": speed,
        "vx_raw": vx_raw,
    }


def compute_pwm_signals(
    t_pwm: np.ndarray,
    pwm_values: np.ndarray,
    t_vel: np.ndarray,
    pwm_names: List[str],
    rotor_constant: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Optional[np.ndarray], Dict[str, np.ndarray]]:
    motor_values = pwm_to_motor_values(pwm_values)
    force_values = convert_motor_values_to_force(motor_values, rotor_constant)
    name_to_idx = {name: idx for idx, name in enumerate(pwm_names)}
    motor_interp_by_thruster: Dict[str, np.ndarray] = {}
    for thruster in PLOT_THRUSTERS:
        if thruster in name_to_idx:
            motor_interp_by_thruster[thruster] = interpolate_series(
                t_pwm, motor_values[:, name_to_idx[thruster]], t_vel
            )
    driver_signal = None
    if all(thruster in motor_interp_by_thruster for thruster in PLOT_THRUSTERS):
        driver_signal = 0.5 * (
            motor_interp_by_thruster[PLOT_THRUSTERS[0]]
            + motor_interp_by_thruster[PLOT_THRUSTERS[1]]
        )
    return motor_values, force_values, name_to_idx, driver_signal, motor_interp_by_thruster


def compute_highlight_segments(
    t_vel: np.ndarray,
    primary_vel: np.ndarray,
    driver_signal: Optional[np.ndarray],
    min_motor: float,
    max_motor_slope: float,
    min_duration: float,
    min_speed: float,
    max_speed_slope: float,
) -> Tuple[
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[Tuple[float, float, float, float]],
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[Tuple[float, float]],
]:
    highlight_segments_all: List[Tuple[float, float]] = []
    fit_lines: List[Tuple[float, float, float, float]] = []
    pwm_events_plot: List[Tuple[float, float]] = []
    pwm_positive_plot: List[Tuple[float, float]] = []
    yellow_segments: List[Tuple[float, float]] = []
    accel_pos_segments: List[Tuple[float, float]] = []
    accel_neg_segments: List[Tuple[float, float]] = []
    if driver_signal is None:
        return (
            pwm_events_plot,
            pwm_positive_plot,
            highlight_segments_all,
            fit_lines,
            yellow_segments,
            accel_pos_segments,
            accel_neg_segments,
        )
    if len(t_vel) < 3:
        return (
            pwm_events_plot,
            pwm_positive_plot,
            highlight_segments_all,
            fit_lines,
            yellow_segments,
            accel_pos_segments,
            accel_neg_segments,
        )
    # Blue windows: non-zero, constant PWM segments.
    pwm_const_mask = constant_mask(
        t_vel,
        driver_signal,
        min_value=min_motor,
        max_slope=max_motor_slope,
        use_abs=True,
    )
    pwm_events = segments_from_mask(t_vel, pwm_const_mask, min_duration=min_duration)
    pwm_events_plot = clip_segments(pwm_events, DEFAULT_HIGHLIGHT_MIN_START)
    pwm_positive_plot = pwm_events_plot.copy()

    # Yellow windows: non-zero, near-constant primary velocity segments.
    vel_const_mask = constant_mask(
        t_vel,
        primary_vel,
        min_value=min_speed,
        max_slope=max_speed_slope,
        use_abs=False,
    )
    vel_const_mask &= primary_vel >= min_speed
    vel_segments = segments_from_mask(t_vel, vel_const_mask, min_duration=min_duration)

    for seg_start, seg_end in vel_segments:
        for blue_start, blue_end in pwm_events:
            if not _segments_overlap((seg_start, seg_end), (blue_start, blue_end)):
                continue
            clip_start = max(seg_start, blue_start)
            clip_end = min(seg_end, blue_end)
            if (clip_end - clip_start) < min_duration:
                continue
            yellow_segments.append((clip_start, clip_end))
            highlight_segments_all.append((clip_start, clip_end))
            fit = linear_fit_segment(t_vel, primary_vel, clip_start, clip_end)
            if fit is not None:
                slope, intercept = fit
                fit_lines.append((slope, clip_start, clip_end, intercept))

    accel = np.gradient(primary_vel, t_vel)
    accel_const_mask = constant_mask(
        t_vel,
        accel,
        min_value=DEFAULT_MIN_AVG_ACCEL,
        max_slope=max_speed_slope,
        use_abs=True,
    )
    accel_segments = segments_from_mask(t_vel, accel_const_mask, min_duration=min_duration)
    for seg_start, seg_end in accel_segments:
        for blue_start, blue_end in pwm_events:
            if not _segments_overlap((seg_start, seg_end), (blue_start, blue_end)):
                continue
            clip_start = max(seg_start, blue_start)
            clip_end = min(seg_end, blue_end)
            if (clip_end - clip_start) < min_duration:
                continue
            fit = linear_fit_segment(t_vel, primary_vel, clip_start, clip_end)
            if fit is None:
                continue
            slope, intercept = fit
            if slope > 0.0:
                accel_pos_segments.append((clip_start, clip_end))
            elif slope < 0.0:
                accel_neg_segments.append((clip_start, clip_end))
            else:
                continue
            highlight_segments_all.append((clip_start, clip_end))
            fit_lines.append((slope, clip_start, clip_end, intercept))

    highlight_segments_all = clip_segments(
        highlight_segments_all, DEFAULT_HIGHLIGHT_MIN_START
    )
    clipped_fit_lines = []
    for slope, s, e, b in fit_lines:
        if e <= DEFAULT_HIGHLIGHT_MIN_START:
            continue
        s_clip = max(s, DEFAULT_HIGHLIGHT_MIN_START)
        clipped_fit_lines.append((slope, s_clip, e, b))
    fit_lines = clipped_fit_lines
    pwm_positive_plot = clip_segments(pwm_positive_plot, DEFAULT_HIGHLIGHT_MIN_START)
    return (
        pwm_events_plot,
        pwm_positive_plot,
        highlight_segments_all,
        fit_lines,
        yellow_segments,
        accel_pos_segments,
        accel_neg_segments,
    )


def find_pwm_events(
    times: np.ndarray,
    signal: np.ndarray,
    min_value: float,
    max_slope: float,
    min_duration: float,
) -> List[Tuple[float, float]]:
    active_mask = np.abs(signal) >= min_value
    active_segments = segments_from_mask(times, active_mask, min_duration=min_duration)
    if not active_segments:
        return []
    steady_mask = constant_mask(
        times,
        signal,
        min_value=min_value,
        max_slope=max_slope,
        use_abs=True,
    )
    steady_segments = segments_from_mask(times, steady_mask, min_duration=min_duration)
    if not steady_segments:
        return []
    keep: List[Tuple[float, float]] = []
    for seg in active_segments:
        if any(_segments_overlap(seg, steady_seg) for steady_seg in steady_segments):
            keep.append(seg)
    return keep


def linear_fit_segment(
    times: np.ndarray,
    values: np.ndarray,
    start_t: float,
    end_t: float,
) -> Optional[Tuple[float, float]]:
    mask = (times >= start_t) & (times <= end_t)
    if np.count_nonzero(mask) < 2:
        return None
    t = times[mask]
    v = values[mask]
    coeffs = np.polyfit(t, v, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    return slope, intercept


def select_linear_rise_segment(
    times: np.ndarray,
    vx: np.ndarray,
    seg_start: float,
    end_t: float,
    min_duration: float,
    min_speed: float,
    pwm_events: List[Tuple[float, float]],
    event_idx: int,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    search_mask = (times >= seg_start) & (times <= end_t)
    idxs = np.where(search_mask)[0]
    if idxs.size < 3:
        return None
    rise_start_t = float(times[idxs[0]])
    required_duration = DEFAULT_MIN_VEL_SPIKE_SEC
    rise_start_v = float(vx[idxs[0]])
    peak_indices = find_local_max_indices(times, vx, idxs[0], end_t)
    if not peak_indices:
        return None
    best_fit = None
    best_score = None
    for peak_idx in peak_indices:
        if vx[peak_idx] < min_speed:
            continue
        rise_end_t = float(times[peak_idx])
        if rise_end_t <= rise_start_t:
            continue
        rise_len = rise_end_t - rise_start_t
        if rise_len < required_duration:
            continue
        avg_slope = (vx[peak_idx] - rise_start_v) / rise_len
        if avg_slope <= DEFAULT_MIN_AVG_ACCEL:
            continue
        rise_seg = (rise_start_t, rise_end_t)
        if any(
            _segments_overlap(rise_seg, other)
            for j, other in enumerate(pwm_events)
            if j != event_idx
        ):
            continue
        fit_start = rise_start_t + rise_len / 6.0
        fit_end = rise_end_t - rise_len / 6.0
        if fit_end <= fit_start:
            continue
        fit = linear_fit_segment(times, vx, fit_start, fit_end)
        if fit is None:
            continue
        slope, intercept = fit
        if slope <= 0.0:
            continue
        score = (slope, rise_len, vx[peak_idx])
        if best_score is None or score > best_score:
            best_score = score
            best_fit = (slope, rise_start_t, rise_end_t, intercept, fit_start, fit_end)
    return best_fit


def find_local_max_indices(
    times: np.ndarray,
    vx: np.ndarray,
    start_idx: int,
    end_t: float,
) -> List[int]:
    if start_idx >= len(times) - 2:
        return []
    end_idx = np.searchsorted(times, end_t, side="right") - 1
    if end_idx <= start_idx + 1:
        return []
    peaks: List[int] = []
    for i in range(start_idx + 1, end_idx):
        if (vx[i] >= vx[i - 1] and vx[i] >= vx[i + 1]) and (
            vx[i] > vx[i - 1] or vx[i] > vx[i + 1]
        ):
            peaks.append(int(i))
    return peaks


def convert_motor_values_to_force(
    motor_values: np.ndarray, rotor_constant: float
) -> np.ndarray:
    return rotor_constant * np.abs(motor_values) * motor_values


def verify_constant_force(
    times: np.ndarray,
    net_force: np.ndarray,
    segments: List[Tuple[float, float]],
    rtol: float,
    atol: float,
    label: str,
) -> None:
    if not segments:
        print(f"[WARN] {label}: no highlighted segments for force constancy check.")
        return
    all_ok = True
    for idx, (seg_start, seg_end) in enumerate(segments, start=1):
        mask = (times >= seg_start) & (times <= seg_end)
        if not np.any(mask):
            continue
        seg = net_force[mask]
        ok = np.allclose(seg, seg[0], rtol=rtol, atol=atol)
        if not ok:
            all_ok = False
            max_dev = float(np.max(np.abs(seg - np.mean(seg))))
            print(
                f"[WARN] {label}: segment {idx} force not constant "
                f"(max deviation {max_dev:.4f} N)."
            )
    if all_ok:
        print(f"[INFO] {label}: net force constant within rtol={rtol}, atol={atol}.")


def analytic_drag_forces(
    vel_b: np.ndarray,
    mass: float,
    inertia: Tuple[float, float, float],
    water_rho: float,
) -> np.ndarray:
    inertia_arr = np.array(inertia, dtype=float)
    ri = np.sqrt((3.0 / (2.0 * mass)) * (np.roll(inertia_arr, 1) + np.roll(inertia_arr, -1) - inertia_arr))
    rj = np.roll(ri, 1)
    rk = np.roll(ri, -1)
    forces = -2.0 * water_rho * rj * rk * np.abs(vel_b) * vel_b
    return forces


def compute_thruster_force_x(
    t_pwm: np.ndarray,
    force_values: np.ndarray,
    name_to_idx: Dict[str, int],
    t_vel: np.ndarray,
    axes: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    force_x: Dict[str, np.ndarray] = {}
    for name, axis in axes.items():
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        interp = interpolate_series(t_pwm, force_values[:, idx], t_vel)
        force_x[name] = interp * float(axis[0])
    return force_x


def collect_front_rear_samples(
    times: np.ndarray,
    highlight_segments: List[Tuple[float, float]],
    fit_lines: List[Tuple[float, float, float, float]],
    rear_force_x: np.ndarray,
    front_force_x: np.ndarray,
    drag_x: np.ndarray,
    mass: float,
) -> List[Tuple[float, float, float, float]]:
    samples: List[Tuple[float, float, float, float]] = []
    for slope, fit_start, fit_end, _ in fit_lines:
        seg = find_overlapping_segment((fit_start, fit_end), highlight_segments)
        if seg is None:
            continue
        seg_start, seg_end = seg
        mask = (times >= seg_start) & (times <= seg_end)
        if not np.any(mask):
            continue
        true_force = float(slope) * float(mass)
        rear_mean = float(np.nanmean(rear_force_x[mask]))
        front_mean = float(np.nanmean(front_force_x[mask]))
        drag_mean = float(np.nanmean(drag_x[mask]))
        samples.append((true_force, rear_mean, front_mean, drag_mean))
    return samples


def fit_front_rear_coeffs(
    samples: List[Tuple[float, float, float, float]]
) -> Optional[Tuple[float, float]]:
    if len(samples) < 2:
        return None
    y = np.array([s[0] - s[3] for s in samples], dtype=float)
    x = np.array([[s[1], s[2]] for s in samples], dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def mse_for_coeffs(
    samples: List[Tuple[float, float, float, float]],
    coeffs: Tuple[float, float],
) -> Optional[float]:
    if not samples:
        return None
    k_rear, k_front = coeffs
    errs = []
    for true_force, rear_force, front_force, drag_force in samples:
        pred = k_rear * rear_force + k_front * front_force + drag_force
        errs.append((pred - true_force) ** 2)
    return float(np.mean(errs)) if errs else None


def load_drag_log(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data:
        records = data["records"]
    else:
        records = data
    times = np.array([r["t"] for r in records], dtype=float)
    forces = np.array([r["drag_force_b"] for r in records], dtype=float)
    return times, forces


def resolve_drag_log_path(base_dir: Path, test_id: int, prefix: str) -> Optional[Path]:
    candidate = base_dir / f"{prefix}{test_id}.json"
    if candidate.exists():
        return candidate
    fallback = base_dir / f"{prefix}.json"
    if fallback.exists():
        return fallback
    return None


def load_drag_force_aligned(
    drag_log_dir: Path,
    test_id: int,
    prefix: str,
    t_vel: np.ndarray,
    label: str,
) -> Optional[np.ndarray]:
    path = resolve_drag_log_path(drag_log_dir, test_id, prefix)
    if path is None:
        print(f"[WARN] {label} drag log not found for test {test_id:04d}.")
        return None
    times, forces = load_drag_log(path)
    force_x = forces if forces.ndim == 1 else forces[:, 0]
    return interpolate_series(times, force_x, t_vel)


def filter_fit_lines(
    fit_lines: List[Tuple[float, float, float, float]],
    segments: List[Tuple[float, float]],
) -> List[Tuple[float, float, float, float]]:
    if not fit_lines or not segments:
        return []
    kept: List[Tuple[float, float, float, float]] = []
    for slope, s, e, b in fit_lines:
        if any(_segments_overlap((s, e), seg) for seg in segments):
            kept.append((slope, s, e, b))
    return kept


def plot_thrust_vs_drags_highlighted(
    title: str,
    times: np.ndarray,
    net_force: np.ndarray,
    cuboid_force: Optional[np.ndarray],
    cfd_force: Optional[np.ndarray],
    segments: List[Tuple[float, float]],
    pwm_positive: List[Tuple[float, float]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    net_plot = np.full_like(times, np.nan, dtype=float)
    cub_plot = np.full_like(times, np.nan, dtype=float) if cuboid_force is not None else None
    cfd_plot = np.full_like(times, np.nan, dtype=float) if cfd_force is not None else None

    mean_labels_added = False
    for seg_start, seg_end in segments:
        mask = (times >= seg_start) & (times <= seg_end)
        if not np.any(mask):
            continue
        net_plot[mask] = net_force[mask]
        if cuboid_force is not None:
            cub_plot[mask] = cuboid_force[mask]
        if cfd_force is not None:
            cfd_plot[mask] = cfd_force[mask]
        ax.axvspan(seg_start, seg_end, color="gold", alpha=0.2, zorder=0)

        event = find_overlapping_segment((seg_start, seg_end), pwm_positive)
        if event is not None:
            event_mask = (times >= event[0]) & (times <= event[1])
        else:
            event_mask = mask
        if not np.any(event_mask):
            print(
                f"[WARN] {title}: no positive PWM overlap for segment "
                f"{seg_start:.2f}-{seg_end:.2f}s; skipping thrust mean."
            )
            continue
        net_mean = float(np.nanmean(net_force[event_mask]))
        net_label = "net mean" if not mean_labels_added else None
        ax.hlines(
            net_mean,
            seg_start,
            seg_end,
            colors="tab:red",
            linestyles="--",
            linewidth=1.0,
            label=net_label,
        )
        mid_t = 0.5 * (seg_start + seg_end)
        ax.annotate(
            f"{net_mean:.2f}N",
            xy=(mid_t, net_mean),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="tab:red",
            fontsize=8,
        )

        if cuboid_force is not None:
            cub_mean = float(np.nanmean(cuboid_force[mask]))
            cub_label = "cuboid mean" if not mean_labels_added else None
            ax.hlines(
                cub_mean,
                seg_start,
                seg_end,
                colors="tab:blue",
                linestyles="--",
                linewidth=1.0,
                label=cub_label,
            )
            ax.annotate(
                f"{cub_mean:.2f}N",
                xy=(mid_t, cub_mean),
                xytext=(0, -6),
                textcoords="offset points",
                ha="center",
                va="top",
                color="tab:blue",
                fontsize=8,
            )

        if cfd_force is not None:
            cfd_mean = float(np.nanmean(cfd_force[mask]))
            cfd_label = "cfd mean" if not mean_labels_added else None
            ax.hlines(
                cfd_mean,
                seg_start,
                seg_end,
                colors="tab:green",
                linestyles="--",
                linewidth=1.0,
                label=cfd_label,
            )
            ax.annotate(
                f"{cfd_mean:.2f}N",
                xy=(mid_t, cfd_mean),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="tab:green",
                fontsize=8,
            )
        mean_labels_added = True

    ax.plot(times, net_plot, label="net thrust (drivers)", color="tab:red", linewidth=1.5)
    if cub_plot is not None:
        ax.plot(times, cub_plot, label="cuboid drag", color="tab:blue", linewidth=1.5)
    if cfd_plot is not None:
        ax.plot(times, cfd_plot, label="cfd drag", color="tab:green", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drag_components(
    title_prefix: str,
    cuboid_t: Optional[np.ndarray],
    cuboid_force: Optional[np.ndarray],
    cfd_t: Optional[np.ndarray],
    cfd_force: Optional[np.ndarray],
    t_vel: Optional[np.ndarray],
    vel_b: Optional[np.ndarray],
) -> List[plt.Figure]:
    if cuboid_t is None and cfd_t is None:
        return []
    figures: List[plt.Figure] = []
    labels = ["Fx", "Fy", "Fz"]
    for idx in range(3):
        fig, ax = plt.subplots(figsize=(10, 4))
        if cuboid_t is not None and cuboid_force is not None:
            ax.plot(
                cuboid_t,
                cuboid_force[:, idx],
                color="tab:blue",
                linewidth=1.2,
                label="cuboid drag",
            )
        if cfd_t is not None and cfd_force is not None:
            ax.plot(
                cfd_t,
                cfd_force[:, idx],
                color="tab:green",
                linewidth=1.2,
                label="cfd drag",
            )

        ax_vel = None
        if (
            t_vel is not None
            and vel_b is not None
            and len(t_vel) > 0
            and vel_b.shape[0] == len(t_vel)
        ):
            ax_vel = ax.twinx()
            ax_vel.plot(
                t_vel,
                vel_b[:, 0],
                color="tab:gray",
                linewidth=1.0,
                linestyle="--",
                label="vxb",
            )
            ax_vel.plot(
                t_vel,
                vel_b[:, 1],
                color="tab:orange",
                linewidth=1.0,
                linestyle="--",
                label="vyb",
            )
            ax_vel.plot(
                t_vel,
                vel_b[:, 2],
                color="tab:purple",
                linewidth=1.0,
                linestyle="--",
                label="vzb",
            )
            ax_vel.set_ylabel("Velocity [m/s]")
            ax_vel.tick_params(axis="y")

        ax.set_ylabel(f"{labels[idx]} [N]")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"{title_prefix} {labels[idx]}")
        ax.grid(True, alpha=0.3)

        handles, labels_legend = ax.get_legend_handles_labels()
        if ax_vel is not None:
            h2, l2 = ax_vel.get_legend_handles_labels()
            handles += h2
            labels_legend += l2
        ax.legend(handles, labels_legend)
        fig.tight_layout()
        figures.append(fig)
    return figures


def plot_speed_force_drag(
    title: str,
    times: np.ndarray,
    speed: np.ndarray,
    net_force: np.ndarray,
    cuboid_force: Optional[np.ndarray],
    cfd_force: Optional[np.ndarray],
    segments: List[Tuple[float, float]],
    pwm_positive: List[Tuple[float, float]],
) -> plt.Figure:
    fig, ax_speed = plt.subplots(figsize=(10, 4))
    ax_force = ax_speed.twinx()

    ax_speed.plot(times, speed, color="black", linewidth=1.2, label="speed")
    ax_force.plot(times, net_force, color="tab:red", linewidth=1.2, label="net thrust")

    if cuboid_force is not None:
        ax_force.plot(times, cuboid_force, color="tab:blue", linewidth=1.2, label="cuboid drag")
    if cfd_force is not None:
        ax_force.plot(times, cfd_force, color="tab:green", linewidth=1.2, label="cfd drag")

    mean_labels_added = False
    for seg_start, seg_end in segments:
        ax_speed.axvspan(seg_start, seg_end, color="gold", alpha=0.2, zorder=0)
        mask = (times >= seg_start) & (times <= seg_end)
        if not np.any(mask):
            continue
        event = find_overlapping_segment((seg_start, seg_end), pwm_positive)
        if event is not None:
            event_mask = (times >= event[0]) & (times <= event[1])
        else:
            event_mask = mask
        if not np.any(event_mask):
            print(
                f"[WARN] {title}: no positive PWM overlap for segment "
                f"{seg_start:.2f}-{seg_end:.2f}s; skipping thrust mean."
            )
            continue
        net_mean = float(np.nanmean(net_force[event_mask]))
        net_label = "net mean" if not mean_labels_added else None
        ax_force.hlines(
            net_mean,
            seg_start,
            seg_end,
            colors="tab:red",
            linestyles="--",
            linewidth=1.0,
            label=net_label,
        )
        mid_t = 0.5 * (seg_start + seg_end)
        ax_force.annotate(
            f"{net_mean:.2f}N",
            xy=(mid_t, net_mean),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="tab:red",
            fontsize=8,
        )

        if cuboid_force is not None:
            cub_mean = float(np.nanmean(cuboid_force[mask]))
            cub_label = "cuboid mean" if not mean_labels_added else None
            ax_force.hlines(
                cub_mean,
                seg_start,
                seg_end,
                colors="tab:blue",
                linestyles="--",
                linewidth=1.0,
                label=cub_label,
            )
            ax_force.annotate(
                f"{cub_mean:.2f}N",
                xy=(mid_t, cub_mean),
                xytext=(0, -6),
                textcoords="offset points",
                ha="center",
                va="top",
                color="tab:blue",
                fontsize=8,
            )
        if cfd_force is not None:
            cfd_mean = float(np.nanmean(cfd_force[mask]))
            cfd_label = "cfd mean" if not mean_labels_added else None
            ax_force.hlines(
                cfd_mean,
                seg_start,
                seg_end,
                colors="tab:green",
                linestyles="--",
                linewidth=1.0,
                label=cfd_label,
            )
            ax_force.annotate(
                f"{cfd_mean:.2f}N",
                xy=(mid_t, cfd_mean),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="tab:green",
                fontsize=8,
            )
        mean_labels_added = True

    ax_speed.set_xlabel("Time [s]")
    ax_speed.set_ylabel("Speed [m/s]")
    ax_force.set_ylabel("Force [N]")
    ax_speed.set_title(title)
    ax_speed.grid(True, alpha=0.3)

    handles_speed, labels_speed = ax_speed.get_legend_handles_labels()
    handles_force, labels_force = ax_force.get_legend_handles_labels()
    ax_speed.legend(handles_speed + handles_force, labels_speed + labels_force, loc="upper right")
    fig.tight_layout()
    return fig


def collect_force_sample_rows(
    times: np.ndarray,
    highlight_segments: List[Tuple[float, float]],
    fit_lines: List[Tuple[float, float, float, float]],
    net_force: Optional[np.ndarray],
    cuboid_force: Optional[np.ndarray],
    cfd_force: Optional[np.ndarray],
    mass: float,
) -> List[Tuple[float, float, Optional[float], Optional[float]]]:
    rows: List[Tuple[float, float, Optional[float], Optional[float]]] = []
    if net_force is None or not fit_lines:
        return rows
    for slope, fit_start, fit_end, _ in fit_lines:
        seg = find_overlapping_segment((fit_start, fit_end), highlight_segments)
        if seg is None:
            continue
        seg_start, seg_end = seg
        mask = (times >= seg_start) & (times <= seg_end)
        if not np.any(mask):
            continue
        true_force = float(slope) * float(mass)
        thr_mean = float(np.nanmean(net_force[mask]))
        cuboid_pred = None
        cfd_pred = None
        if cuboid_force is not None:
            cub_mean = float(np.nanmean(cuboid_force[mask]))
            cuboid_pred = thr_mean + cub_mean
        if cfd_force is not None:
            cfd_mean = float(np.nanmean(cfd_force[mask]))
            cfd_pred = thr_mean + cfd_mean
        rows.append((true_force, thr_mean, cuboid_pred, cfd_pred))
    return rows


def plot_force_distribution(
    true_forces: List[float],
    pred_cuboid: List[float],
    pred_cfd: List[float],
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    data_sets = [
        ("true net force (m*a)", true_forces, "tab:blue"),
        ("pred net force (thruster + cuboid drag)", pred_cuboid, "tab:orange"),
        ("pred net force (thruster + CFD drag)", pred_cfd, "tab:green"),
    ]
    all_vals = np.array(true_forces + pred_cuboid + pred_cfd, dtype=float)
    bins = np.histogram_bin_edges(all_vals, bins="auto") if all_vals.size else 10

    for ax, (label, values, color) in zip(axes, data_sets):
        if values:
            ax.hist(values, bins=bins, density=True, alpha=0.6, color=color)
        else:
            ax.text(
                0.5,
                0.5,
                "No samples",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="tab:gray",
            )
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Force [N]")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_force_distribution_calibrated(
    true_forces: List[float],
    pred_cuboid: List[float],
    pred_cfd: List[float],
    coeff_cub: Optional[object],
    coeff_cfd: Optional[object],
    title: str,
) -> plt.Figure:
    fig = plot_force_distribution(true_forces, pred_cuboid, pred_cfd, title)
    def _format_coeff(label: str, coeff: object) -> Optional[str]:
        if coeff is None:
            return None
        if isinstance(coeff, (tuple, list)):
            if len(coeff) == 2:
                a, b = coeff
                return f"{label} a={a:.3g}, b={b:.3g}"
            if len(coeff) == 3:
                a, b, c = coeff
                return f"{label} a={a:.3g}, b={b:.3g}, c={c:.3g}"
        try:
            return f"{label} k={float(coeff):.3f}"
        except (TypeError, ValueError):
            return None

    coeff_lines = []
    line = _format_coeff("cuboid", coeff_cub)
    if line:
        coeff_lines.append(line)
    line = _format_coeff("cfd", coeff_cfd)
    if line:
        coeff_lines.append(line)
    if coeff_lines:
        ax = fig.axes[0]
        ax.text(
            0.98,
            0.02,
            "\n".join(coeff_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def compute_mse(pairs: List[Tuple[float, float]]) -> Optional[float]:
    if not pairs:
        return None
    diffs = [(pred - true) ** 2 for true, pred in pairs]
    return float(np.mean(diffs)) if diffs else None


def plot_force_comparison(
    true_cub_pairs: List[Tuple[float, float]],
    true_cfd_pairs: List[Tuple[float, float]],
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    if not true_cub_pairs and not true_cfd_pairs:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No force samples available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.tight_layout()
        return fig

    if true_cub_pairs:
        x_cub, y_cub = zip(*true_cub_pairs)
        ax.scatter(
            x_cub,
            y_cub,
            s=20,
            alpha=0.7,
            color="tab:orange",
            label="cuboid drag",
        )
    if true_cfd_pairs:
        x_cfd, y_cfd = zip(*true_cfd_pairs)
        ax.scatter(
            x_cfd,
            y_cfd,
            s=20,
            alpha=0.7,
            color="tab:green",
            label="CFD drag",
        )

    all_vals = []
    if true_cub_pairs:
        all_vals.extend([v for pair in true_cub_pairs for v in pair])
    if true_cfd_pairs:
        all_vals.extend([v for pair in true_cfd_pairs for v in pair])
    if all_vals:
        min_v = float(np.min(all_vals))
        max_v = float(np.max(all_vals))
        pad = 0.05 * (max_v - min_v) if max_v > min_v else 1.0
        lo = min_v - pad
        hi = max_v + pad
        ax.plot([lo, hi], [lo, hi], color="tab:blue", linestyle="--", linewidth=1.0, label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("True net force (m*a) [N]")
    ax.set_ylabel("Predicted net force [N]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def fit_thruster_scale(
    rows: List[Tuple[float, float, Optional[float], Optional[float]]],
    drag_index: int,
) -> Optional[float]:
    num = 0.0
    den = 0.0
    for row in rows:
        true_force = row[0]
        thr_mean = row[1]
        drag_pred = row[drag_index]
        if drag_pred is None or not np.isfinite(thr_mean):
            continue
        drag_mean = drag_pred - thr_mean
        target = true_force - drag_mean
        num += thr_mean * target
        den += thr_mean * thr_mean
    if den <= 0.0:
        return None
    return num / den


def fit_thruster_quadratic(
    rows: List[Tuple[float, float, Optional[float], Optional[float]]],
    drag_index: int,
) -> Optional[Tuple[float, float]]:
    xs = []
    ys = []
    for row in rows:
        true_force = row[0]
        thr_mean = row[1]
        drag_pred = row[drag_index]
        if drag_pred is None or not np.isfinite(thr_mean):
            continue
        drag_mean = drag_pred - thr_mean
        target = true_force - drag_mean
        xs.append(thr_mean)
        ys.append(target)
    if len(xs) < 2:
        return None
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    design = np.column_stack((x_arr**2, x_arr))
    coeffs, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def plot_force_comparison_with_coeffs(
    true_cub_pairs: List[Tuple[float, float]],
    true_cfd_pairs: List[Tuple[float, float]],
    coeff_cub: Optional[float],
    coeff_cfd: Optional[float],
    title: str,
) -> plt.Figure:
    fig = plot_force_comparison(true_cub_pairs, true_cfd_pairs, title)
    ax = fig.axes[0]
    coeff_lines = []
    if coeff_cub is not None:
        coeff_lines.append(f"cuboid k={coeff_cub:.3f}")
    if coeff_cfd is not None:
        coeff_lines.append(f"cfd k={coeff_cfd:.3f}")
    if coeff_lines:
        ax.text(
            0.98,
            0.02,
            "\n".join(coeff_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
    fig.tight_layout()
    return fig


def plot_force_comparison_with_quadratic(
    true_cub_pairs: List[Tuple[float, float]],
    true_cfd_pairs: List[Tuple[float, float]],
    coeff_cub: Optional[Tuple[float, float]],
    coeff_cfd: Optional[Tuple[float, float]],
    title: str,
) -> plt.Figure:
    fig = plot_force_comparison(true_cub_pairs, true_cfd_pairs, title)
    ax = fig.axes[0]
    coeff_lines = []
    if coeff_cub is not None:
        a, b = coeff_cub
        coeff_lines.append(f"cuboid a={a:.3g}, b={b:.3g}")
    if coeff_cfd is not None:
        a, b = coeff_cfd
        coeff_lines.append(f"cfd a={a:.3g}, b={b:.3g}")
    if coeff_lines:
        ax.text(
            0.98,
            0.02,
            "\n".join(coeff_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
        fig.tight_layout()
    return fig


def set_equal_axes(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    max_range = (maxs - mins).max()
    if not np.isfinite(max_range) or max_range == 0.0:
        max_range = 1.0
    mids = 0.5 * (maxs + mins)
    ax.set_xlim(mids[0] - max_range / 2, mids[0] + max_range / 2)
    ax.set_ylim(mids[1] - max_range / 2, mids[1] + max_range / 2)
    ax.set_zlim(mids[2] - max_range / 2, mids[2] + max_range / 2)


def plot_trajectory_with_vectors(
    test_id: int,
    positions_b: np.ndarray,
    t_vel: np.ndarray,
    vel_b: np.ndarray,
    highlight_segments_by_thruster: List[List[Tuple[float, float]]],
) -> plt.Figure:
    if HAS_MPL_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(positions_b[:, 0], positions_b[:, 1], positions_b[:, 2], label="trajectory", linewidth=2)
        ax.scatter(positions_b[0, 0], positions_b[0, 1], positions_b[0, 2], s=50, marker="o", label="start")
        ax.set_xlabel("Xb [m]")
        ax.set_ylabel("Yb [m]")
        ax.set_zlabel("Zb [m]")
        ax.set_title(f"{MODE_LABEL} Test{test_id:04d} Trajectory + Body-Frame Velocity")
        ax.grid(True)

        pos_for_vel = positions_b[:-1]
        for thruster, segments in zip(PLOT_THRUSTERS, highlight_segments_by_thruster):
            color = VECTOR_COLORS.get(thruster, "tab:gray")
            for seg_start, seg_end in segments:
                mask = (t_vel >= seg_start) & (t_vel <= seg_end)
                if not np.any(mask):
                    continue
                ax.quiver(
                    pos_for_vel[mask, 0],
                    pos_for_vel[mask, 1],
                    pos_for_vel[mask, 2],
                    vel_b[mask, 0],
                    vel_b[mask, 1],
                    vel_b[mask, 2],
                    color=color,
                    linewidth=0.8,
                    alpha=0.9,
                )

        legend_handles = [
            Line2D([0], [0], color="tab:blue", lw=2, label="trajectory"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="k", label="start", markersize=6),
        ]
        for thruster in PLOT_THRUSTERS:
            legend_handles.append(
                Line2D([0], [0], color=VECTOR_COLORS.get(thruster, "tab:gray"), lw=2, label=f"{thruster} vectors")
            )
        ax.legend(handles=legend_handles, loc="upper right")
        set_equal_axes(ax, positions_b)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(positions_b[:, 0], positions_b[:, 1], label="trajectory", linewidth=2)
    ax.scatter(positions_b[0, 0], positions_b[0, 1], s=40, marker="o", label="start")
    ax.set_xlabel("Xb [m]")
    ax.set_ylabel("Yb [m]")
    ax.set_title(f"{MODE_LABEL} Test{test_id:04d} Trajectory + Body-Frame Velocity (XY)")
    ax.grid(True)

    pos_for_vel = positions_b[:-1]
    for thruster, segments in zip(PLOT_THRUSTERS, highlight_segments_by_thruster):
        color = VECTOR_COLORS.get(thruster, "tab:gray")
        for seg_start, seg_end in segments:
            mask = (t_vel >= seg_start) & (t_vel <= seg_end)
            if not np.any(mask):
                continue
            ax.quiver(
                pos_for_vel[mask, 0],
                pos_for_vel[mask, 1],
                vel_b[mask, 0],
                vel_b[mask, 1],
                color=color,
                linewidth=0.8,
                alpha=0.9,
                angles="xy",
                scale_units="xy",
                scale=1.0,
            )

    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def parse_test_id(path: Path) -> Optional[int]:
    match = re.search(r"Curee Test(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def collect_tests(base_dir: Path, run_id: Optional[int]) -> List[Tuple[int, Path]]:
    tests = []
    for path in base_dir.glob("Curee Test*.json"):
        test_id = parse_test_id(path)
        if test_id is None:
            continue
        if run_id is not None and test_id != run_id:
            continue
        tests.append((test_id, path))
    return sorted(tests, key=lambda item: item[0])


def parse_tests_arg(tests_arg: str) -> Optional[List[int]]:
    if tests_arg.lower() == "all":
        return None
    parts = [p.strip() for p in tests_arg.split(",") if p.strip()]
    ids: List[int] = []
    for part in parts:
        try:
            val = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid test id '{part}'. Use comma-separated ints or 'all'.") from exc
        if val < 1:
            raise ValueError(f"Test id {val} out of range. Must be >= 1.")
        ids.append(val)
    return sorted(set(ids))


def load_start_times(prefix: str) -> Dict[int, float]:
    if not START_TIMES_JSON.exists():
        return {}
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
            if not np.isfinite(val):
                continue
            out[run_id] = val
    return out


def discover_test_ids(base_dir: Path) -> List[int]:
    tests = []
    for path in base_dir.glob("Curee Test*.json"):
        test_id = parse_test_id(path)
        if test_id is None:
            continue
        tests.append(test_id)
    return sorted(set(tests))


def discover_run_ids_from_bags(base_dir: Path, bag_prefix: str) -> List[int]:
    runs = []
    for path in base_dir.glob(f"{bag_prefix}*.bag"):
        match = re.search(rf"{bag_prefix}(\d+)", path.name)
        if match:
            runs.append(int(match.group(1)))
    return sorted(set(runs))


def build_run_mapping_ordered(
    base_dir: Path,
    bag_prefix: str,
    excluded_tests: Optional[set] = None,
) -> Dict[int, int]:
    tests = discover_test_ids(base_dir)
    runs = discover_run_ids_from_bags(base_dir, bag_prefix)
    if len(runs) != len(tests):
        raise ValueError(
            f"Run/test count mismatch in {base_dir}: {len(runs)} runs vs {len(tests)} tests."
        )
    mapping = {run_id: test_id for run_id, test_id in zip(runs, tests)}
    if excluded_tests:
        mapping = {
            run_id: test_id
            for run_id, test_id in mapping.items()
            if test_id not in excluded_tests
        }
    if not mapping:
        label = sorted(excluded_tests) if excluded_tests else []
        raise ValueError(f"No valid runs remain after excluding {label}.")
    return mapping


def find_bag_path(base_dir: Path, bag_prefix: str, run_id: int) -> Path:
    matches = sorted(base_dir.glob(f"{bag_prefix}{run_id}*.bag"))
    if not matches:
        raise FileNotFoundError(
            f"No bag found for {bag_prefix}{run_id} in {base_dir}."
        )
    return matches[0]


def build_run_mapping(base_dir: Path, bag_prefix: str) -> Dict[int, int]:
    tests = discover_test_ids(base_dir)
    runs = discover_run_ids_from_bags(base_dir, bag_prefix)
    if len(runs) != len(tests):
        raise ValueError(
            f"Run/test count mismatch in {base_dir}: {len(runs)} runs vs {len(tests)} tests."
        )
    return {run_id: test_id for run_id, test_id in zip(runs, tests)}


def build_test_to_bag(base_dir: Path, bag_prefix: str) -> Dict[int, Path]:
    mapping = build_run_mapping(base_dir, bag_prefix)
    test_to_bag: Dict[int, Path] = {}
    for bag_path in base_dir.glob(f"{bag_prefix}*.bag"):
        match = re.search(rf"{bag_prefix}(\d+)", bag_path.name)
        if not match:
            continue
        run_id = int(match.group(1))
        if run_id not in mapping:
            continue
        test_id = mapping[run_id]
        test_to_bag[test_id] = bag_path
    return test_to_bag


def load_pwm_series(
    bag_path: Path, topic: str, preferred_order: Tuple[str, ...]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    bag = rosbag.Bag(str(bag_path))
    try:
        conn = None
        for info in bag._connections.values():
            if info.topic == topic:
                conn = info
                break
        if conn is None:
            raise ValueError(f"Topic '{topic}' not found in {bag_path.name}")

        msg_classes = generate_dynamic("warpauv_msgs/MotorCommandList", conn.msg_def)
        msg_cls = msg_classes["warpauv_msgs/MotorCommandList"]

        times = []
        values_by_name: Dict[str, List[float]] = {}
        names: List[str] = []
        t0 = None

        for _, raw_msg, stamp in bag.read_messages(topics=[topic], raw=True):
            data = raw_msg[1]
            msg = msg_cls()
            msg.deserialize(data)
            if not msg.motor_commands:
                continue

            if t0 is None:
                t0 = stamp.to_sec()
                seen = [mc.name for mc in msg.motor_commands]
                if all(name in seen for name in preferred_order):
                    names = list(preferred_order)
                else:
                    names = list(seen)
                for name in names:
                    values_by_name[name] = []

            times.append(stamp.to_sec() - t0)
            current = {mc.name: mc.position for mc in msg.motor_commands}
            for name in names:
                val = float(current.get(name, np.nan))
                if not np.isnan(val) and abs(val) < DEADBAND:
                    val = 0.0
                values_by_name[name].append(val)

        if not times:
            raise ValueError(f"No PWM messages found on '{topic}' in {bag_path.name}")

        time_arr = np.array(times, dtype=float)
        values = np.vstack([values_by_name[name] for name in names]).T
        return time_arr, names, values
    finally:
        bag.close()


def pwm_to_motor_values(pwm: np.ndarray) -> np.ndarray:
    motor = np.array(pwm, dtype=float, copy=True)
    finite = np.isfinite(motor)
    motor[finite & (np.abs(motor) < DEADBAND)] = 0.0

    mask_fwd = motor >= DEADBAND
    motor[mask_fwd] = FORWARD_SCALE * (
        FORWARD_A * motor[mask_fwd] ** 2 + FORWARD_B * motor[mask_fwd] + FORWARD_C
    )

    if motor.shape[1] >= 2:
        sub = motor[:, 0:2].copy()
        mask_fwd_12 = sub > DEADBAND
        sub[mask_fwd_12] *= FORWARD_BIAS

        mask_back_12 = sub <= -DEADBAND
        sub[mask_back_12] = BACK_BIAS * (
            BACK_A * sub[mask_back_12] ** 2 + BACK_B * sub[mask_back_12] + BACK_C
        )
        motor[:, 0:2] = sub

    if motor.shape[1] >= 3:
        sub = motor[:, 2:6].copy()
        mask_back_36 = sub <= -DEADBAND
        sub[mask_back_36] = (
            BACK_A * sub[mask_back_36] ** 2 + BACK_B * sub[mask_back_36] + BACK_C
        )
        motor[:, 2:6] = sub

    return motor


def trim_time_series(
    times: np.ndarray, values: np.ndarray, start_time: float
) -> Tuple[np.ndarray, np.ndarray]:
    if start_time <= 0.0:
        return times, values
    mask = times >= start_time
    if not np.any(mask):
        return np.array([], dtype=float), values[:0]
    return times[mask] - start_time, values[mask]


def plot_velocity_pwm(
    test_id: int,
    t_vel: np.ndarray,
    velocity: np.ndarray,
    speed: np.ndarray,
    t_pwm: np.ndarray,
    motor_values: np.ndarray,
    pwm_names: List[str],
    highlight_segments_by_thruster: List[List[Tuple[float, float]]],
    pwm_event_segments: List[Tuple[float, float]],
    yellow_segments: List[Tuple[float, float]],
    accel_pos_segments: List[Tuple[float, float]],
    accel_neg_segments: List[Tuple[float, float]],
    fit_lines: Optional[List[Tuple[float, float, float, float]]],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12, 4.5))
    axes = axes.ravel()

    name_to_idx = {name: idx for idx, name in enumerate(pwm_names)}
    handles = None
    labels = None

    highlight_patch = None
    pos_patch = None
    neg_patch = None
    pwm_patch = None

    for ax, thruster, highlight_segments in zip(
        axes, PLOT_THRUSTERS, highlight_segments_by_thruster
    ):
        ax.plot(t_vel, velocity[:, 0], label="vxb", color="tab:blue", linewidth=1.0)
        ax.plot(t_vel, velocity[:, 1], label="vyb", color="tab:orange", linewidth=1.0)
        ax.plot(t_vel, velocity[:, 2], label="vzb", color="tab:green", linewidth=1.0)
        ax.plot(t_vel, speed, label="speed", color="black", linewidth=1.2)
        ax.set_title(thruster)
        ax.set_ylabel("Velocity [m/s] (body)")
        ax.grid(True, alpha=0.3)

        for seg_start, seg_end in pwm_event_segments:
            patch = ax.axvspan(seg_start, seg_end, color="lightskyblue", alpha=0.18, zorder=0)
            if pwm_patch is None:
                pwm_patch = patch

        for seg_start, seg_end in yellow_segments:
            patch = ax.axvspan(seg_start, seg_end, color="gold", alpha=0.28, zorder=0)
            if highlight_patch is None:
                highlight_patch = patch
        for seg_start, seg_end in accel_pos_segments:
            patch = ax.axvspan(seg_start, seg_end, color="tab:green", alpha=0.22, zorder=0)
            if pos_patch is None:
                pos_patch = patch
        for seg_start, seg_end in accel_neg_segments:
            patch = ax.axvspan(seg_start, seg_end, color="tab:purple", alpha=0.22, zorder=0)
            if neg_patch is None:
                neg_patch = patch

        if fit_lines:
            for slope, seg_start, seg_end, intercept in fit_lines:
                ax.plot(
                    [seg_start, seg_end],
                    [slope * seg_start + intercept, slope * seg_end + intercept],
                    color="tab:gray",
                    linestyle="--",
                    linewidth=1.2,
                    label="avg accel fit" if handles is None else None,
                )
                mid_t = 0.5 * (seg_start + seg_end)
                ax.annotate(
                    f"a={slope:.2f} m/s^2",
                    xy=(mid_t, slope * mid_t + intercept),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="tab:gray",
                    fontsize=8,
                )

        ax2 = ax.twinx()
        if thruster in name_to_idx:
            idx = name_to_idx[thruster]
            line = ax2.plot(
                t_pwm,
                motor_values[:, idx],
                label=f"{thruster} rad/s",
                color="tab:red",
                linewidth=1.0,
                alpha=0.8,
            )
            if handles is None:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                handles = h1 + h2
                labels = l1 + l2
        else:
            ax2.text(
                0.5,
                0.5,
                "PWM missing",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="tab:red",
            )
        ax2.set_ylabel("Motor rad/s")

    for ax in axes:
        ax.set_xlabel("Time [s]")

    fig.suptitle(f"{MODE_LABEL} Test{test_id:04d} Body-Frame Velocity vs Motor rad/s")
    if handles is None:
        h1, l1 = axes[0].get_legend_handles_labels()
        handles, labels = h1, l1
    if handles and labels:
        if pwm_patch is not None:
            handles = handles + [pwm_patch]
            labels = labels + ["pwm>0 event"]
        if highlight_patch is not None:
            handles = handles + [highlight_patch]
            labels = labels + ["constant velocity window"]
        if pos_patch is not None:
            handles = handles + [pos_patch]
            labels = labels + ["positive slope window"]
        if neg_patch is not None:
            handles = handles + [neg_patch]
            labels = labels + ["negative slope window"]
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> None:
    global BASE_DIR, START_TIMES_JSON, BAG_PREFIX, QUALY_PREFIX, MITRE_PREFIX, MODE_LABEL
    parser = argparse.ArgumentParser(
        description="Plot Left/Up Curee test velocity with left thruster motor rad/s."
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
        "--LeftOn",
        action="store_true",
        help="Use MITRE/Left data (default: MITRE/Up).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run id (bag index) to plot; maps to Curee Test number automatically.",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default="all",
        help="Comma-separated run ids or 'all' (default). Overrides --run-id.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=PWM_TOPIC,
        help=f"PWM command topic (default: {PWM_TOPIC})",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=DEFAULT_MIN_SPEED,
        help="Minimum speed (m/s) to consider non-zero.",
    )
    parser.add_argument(
        "--max-speed-slope",
        type=float,
        default=DEFAULT_MAX_SPEED_SLOPE,
        help="Maximum |d(vel)/dt| for near-constant velocity windows.",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=DEFAULT_MIN_SEGMENT_SEC,
        help="Minimum duration (s) for highlighted segments.",
    )
    parser.add_argument(
        "--min-motor",
        type=float,
        default=DEFAULT_MIN_MOTOR,
        help="Minimum |motor| (rad/s) to consider non-zero.",
    )
    parser.add_argument(
        "--max-motor-slope",
        type=float,
        default=DEFAULT_MAX_MOTOR_SLOPE,
        help="Maximum |d(motor)/dt| (rad/s^2) for near-constant motor speed.",
    )
    parser.add_argument(
        "--rotor-constant",
        type=float,
        default=DEFAULT_ROTOR_CONSTANT,
        help="Rotor constant used for force conversion (matches plotMITREPWM).",
    )
    parser.add_argument(
        "--force-const-rtol",
        type=float,
        default=DEFAULT_FORCE_CONST_RTOL,
        help="Relative tolerance for constant net force verification.",
    )
    parser.add_argument(
        "--force-const-atol",
        type=float,
        default=DEFAULT_FORCE_CONST_ATOL,
        help="Absolute tolerance for constant net force verification.",
    )
    parser.add_argument(
        "--water-rho",
        type=float,
        default=DEFAULT_WATER_RHO,
        help="Water density for analytic drag model (kg/m^3).",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=DEFAULT_MASS,
        help="Vehicle mass for analytic drag model (kg).",
    )
    parser.add_argument(
        "--inertia",
        type=float,
        nargs=3,
        default=DEFAULT_INERTIA,
        metavar=("IX", "IY", "IZ"),
        help="Inertia tensor diagonal for analytic drag model.",
    )
    parser.add_argument(
        "--drag-log-cuboid-prefix",
        type=str,
        default=DEFAULT_DRAG_LOG_CUBOID_PREFIX,
        help="Prefix for Cuboid drag log JSON files.",
    )
    parser.add_argument(
        "--drag-log-cfd-prefix",
        type=str,
        default=DEFAULT_DRAG_LOG_CFD_PREFIX,
        help="Prefix for CFD drag log JSON files.",
    )
    parser.add_argument(
        "--drag-log-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory containing drag log JSON files.",
    )
    args = parser.parse_args()
    if args.LeftOn:
        BASE_DIR = LEFT_DIR
        START_TIMES_JSON = LEFT_DIR / "startingLeftTimes.json"
        BAG_PREFIX = "mitreLeft"
        QUALY_PREFIX = "qualyLeft"
        MITRE_PREFIX = "mitreLeft"
        MODE_LABEL = "Left"
    else:
        BASE_DIR = UP_DIR
        START_TIMES_JSON = UP_DIR / "startingUpTimes.json"
        BAG_PREFIX = "mitreUp"
        QUALY_PREFIX = "qualyUp"
        MITRE_PREFIX = "mitreUp"
        MODE_LABEL = "Up"

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    qualy_starts = load_start_times(prefix=QUALY_PREFIX)
    mitre_starts = load_start_times(prefix=MITRE_PREFIX)

    tests_filter = None
    try:
        tests_filter = parse_tests_arg(args.tests)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    try:
        if args.LeftOn:
            run_to_test = build_run_mapping_ordered(
                BASE_DIR, BAG_PREFIX, excluded_tests=LEFT_EXCLUDED_TEST_IDS
            )
        else:
            run_to_test = build_run_mapping_ordered(
                BASE_DIR, BAG_PREFIX, excluded_tests=UP_EXCLUDED_TEST_IDS
            )
    except ValueError as exc:
        print(f"[WARN] {exc}")
        return

    run_ids = sorted(run_to_test.keys())

    if args.run_id is not None:
        if args.run_id not in run_to_test:
            valid = ", ".join(str(r) for r in sorted(run_to_test.keys()))
            print(f"[WARN] No test mapped for run id {args.run_id}. Valid runs: {valid}")
            return
        run_ids = [args.run_id]
    if tests_filter is not None:
        run_ids = [run_id for run_id in tests_filter if run_id in run_to_test]
    if not run_ids:
        print("[WARN] No matching runs after filtering.")
        return

    show_plots = not args.no_show
    drag_log_dir = args.drag_log_dir
    if drag_log_dir == UP_DIR and args.LeftOn:
        drag_log_dir = LEFT_DIR
    thruster_axes = thruster_axes_body()
    all_force_samples: List[Tuple[float, float, float, float]] = []

    for run_id in run_ids:
        test_id = run_to_test[run_id]
        gt_path = BASE_DIR / f"Curee Test{test_id:04d}.json"
        try:
            source_type, source_name, positions, freq_hz = load_positions(gt_path)
        except Exception as exc:
            print(f"[WARN] Skipping {gt_path}: {exc}")
            continue

        start_gt = qualy_starts.get(run_id, 0.0)
        if run_id not in qualy_starts:
            print(f"[WARN] Missing {QUALY_PREFIX} start for run {run_id}; using 0.0s")

        try:
            bag_path = find_bag_path(BASE_DIR, BAG_PREFIX, run_id)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}; skipping PWM overlay.")
            continue

        try:
            t_pwm, pwm_names, pwm_values = load_pwm_series(bag_path, args.topic, THRUSTER_ORDER)
        except Exception as exc:
            print(f"[WARN] {bag_path.name}: {exc}")
            continue

        start_mitre = mitre_starts.get(run_id, 0.0)
        if run_id not in mitre_starts:
            print(f"[WARN] Missing {MITRE_PREFIX} start for run {run_id}; using 0.0s")

        gt_dur = ground_truth_duration_seconds(gt_path)
        pwm_dur = pwm_duration_seconds(t_pwm)
        gt_avail = max(0.0, gt_dur - start_gt)
        pwm_avail = max(0.0, pwm_dur - start_mitre)
        plot_window = max(0.0, min(gt_avail, pwm_avail))

        if plot_window <= 0.0:
            print(f"[WARN] No overlapping time window for test {test_id:04d}; skipping.")
            continue

        positions_trim = trim_by_time(positions, freq_hz, start_gt)
        positions_trim = limit_by_time(positions_trim, freq_hz, plot_window)
        if positions_trim.shape[0] < 3:
            print(f"[WARN] {gt_path} has fewer than 3 samples; skipping.")
            continue

        t_pwm, pwm_values = trim_time_series(t_pwm, pwm_values, start_mitre)
        t_pwm, pwm_values = limit_time_series(t_pwm, pwm_values, plot_window)

        if len(t_pwm) == 0:
            print(f"[WARN] No PWM data after trimming for test {test_id:04d}; skipping.")
            continue

        direction_w = overall_direction(positions_trim)
        primary_dir = primary_direction_for_mode(MODE_LABEL)
        q_wb = quat_from_two_vectors(primary_dir, direction_w)
        positions_rel = positions_trim - positions_trim[0]
        positions_b = rotate_world_to_body(positions_rel, q_wb)
        vel_frames = compute_velocity_frames(positions_trim, freq_hz, q_wb)
        if vel_frames is None:
            print(f"[WARN] {gt_path} has fewer than 2 velocity samples; skipping.")
            continue
        t_vel = np.arange(vel_frames["velocity_raw"].shape[0]) / float(freq_hz)
        primary_vel = primary_velocity_component(vel_frames["vel_b_plot"], MODE_LABEL)

        (
            motor_values,
            force_values,
            name_to_idx,
            driver_signal,
            _,
        ) = compute_pwm_signals(t_pwm, pwm_values, t_vel, pwm_names, args.rotor_constant)
        if driver_signal is None:
            print(f"[WARN] Missing left thruster signals for test {test_id:04d}; skipping highlights.")

        (
            pwm_events_plot,
            pwm_positive_plot,
            highlight_segments_all,
            fit_lines,
            yellow_segments,
            accel_pos_segments,
            accel_neg_segments,
        ) = compute_highlight_segments(
            t_vel,
            primary_vel,
            driver_signal,
            min_motor=args.min_motor,
            max_motor_slope=args.max_motor_slope,
            min_duration=args.min_segment_sec,
            min_speed=args.min_speed,
            max_speed_slope=args.max_speed_slope,
        )
        highlight_segments = highlight_segments_all.copy()

        cuboid_force = load_drag_force_aligned(
            drag_log_dir, test_id, args.drag_log_cuboid_prefix, t_vel, "Cuboid"
        )
        cfd_force = load_drag_force_aligned(
            drag_log_dir, test_id, args.drag_log_cfd_prefix, t_vel, "CFD"
        )
        fit_lines = filter_fit_lines(fit_lines, highlight_segments)

        highlight_segments_by_thruster = [highlight_segments.copy() for _ in PLOT_THRUSTERS]

        force_x_by_thruster = compute_thruster_force_x(
            t_pwm, force_values, name_to_idx, t_vel, thruster_axes
        )
        rear_force_x = None
        front_force_x = None
        if "rear_left" in force_x_by_thruster and "rear_right" in force_x_by_thruster:
            rear_force_x = force_x_by_thruster["rear_left"] + force_x_by_thruster["rear_right"]
        else:
            print(f"[WARN] Missing rear thrusters in PWM names for test {test_id:04d}.")
        if "front_left" in force_x_by_thruster and "front_right" in force_x_by_thruster:
            front_force_x = force_x_by_thruster["front_left"] + force_x_by_thruster["front_right"]
        else:
            print(f"[WARN] Missing front thrusters in PWM names for test {test_id:04d}.")
        net_force = None
        if rear_force_x is not None and front_force_x is not None:
            net_force = rear_force_x + front_force_x

        drag_x = analytic_drag_forces(
            vel_frames["vel_b_plot"], args.mass, args.inertia, args.water_rho
        )[:, 0]

        if net_force is not None and highlight_segments:
            verify_constant_force(
                t_vel,
                net_force,
                highlight_segments,
                rtol=args.force_const_rtol,
                atol=args.force_const_atol,
                label=f"Test{test_id:04d}",
            )

        print(
            f"[INFO] Run{run_id:02d} -> Test{test_id:04d}: {source_type} '{source_name}', "
            f"{len(positions)} samples @ {freq_hz:.1f} Hz"
        )
        fig = plot_velocity_pwm(
            test_id,
            t_vel,
            vel_frames["vel_b_plot"],
            vel_frames["speed"],
            t_pwm,
            motor_values,
            pwm_names,
            highlight_segments_by_thruster,
            pwm_positive_plot,
            yellow_segments,
            accel_pos_segments,
            accel_neg_segments,
            fit_lines,
        )
        fig_traj = plot_trajectory_with_vectors(
            test_id,
            positions_b,
            t_vel,
            vel_frames["vel_b_plot"],
            highlight_segments_by_thruster,
        )

        if (
            rear_force_x is not None
            and front_force_x is not None
            and highlight_segments
            and fit_lines
        ):
            samples = collect_front_rear_samples(
                t_vel,
                highlight_segments,
                fit_lines,
                rear_force_x,
                front_force_x,
                drag_x,
                args.mass,
            )
            all_force_samples.extend(samples)

        fig_drag_highlight = None
        fig_speed_force = None
        fig_drag_components: List[plt.Figure] = []
        if net_force is not None:
            fig_speed_force = plot_speed_force_drag(
                f"{MODE_LABEL} Test{test_id:04d} Speed + Forces",
                t_vel,
                vel_frames["speed"],
                net_force,
                cuboid_force,
                cfd_force,
                highlight_segments,
                pwm_positive_plot,
            )
            if highlight_segments:
                fig_drag_highlight = plot_thrust_vs_drags_highlighted(
                    f"{MODE_LABEL} Test{test_id:04d} Highlighted Thrust vs Drag",
                    t_vel,
                    net_force,
                    cuboid_force,
                    cfd_force,
                    highlight_segments,
                    pwm_positive_plot,
                )
            else:
                print(f"[WARN] No highlight segments for drag plots on test {test_id:04d}.")
        else:
            print(f"[WARN] No net thrust available for drag plots on test {test_id:04d}.")

        cuboid_raw_t = None
        cuboid_raw_f = None
        cfd_raw_t = None
        cfd_raw_f = None
        cuboid_path = resolve_drag_log_path(
            drag_log_dir, test_id, args.drag_log_cuboid_prefix
        )
        if cuboid_path is not None:
            with cuboid_path.open("r") as f:
                data = json.load(f)
            records = data["records"] if isinstance(data, dict) and "records" in data else data
            cuboid_raw_t = np.array([r["t"] for r in records], dtype=float)
            cuboid_raw_f = np.array([r["drag_force_b"] for r in records], dtype=float)
        else:
            print(f"[WARN] Cuboid drag log not found for test {test_id:04d}.")
        cfd_path = resolve_drag_log_path(
            drag_log_dir, test_id, args.drag_log_cfd_prefix
        )
        if cfd_path is not None:
            with cfd_path.open("r") as f:
                data = json.load(f)
            records = data["records"] if isinstance(data, dict) and "records" in data else data
            cfd_raw_t = np.array([r["t"] for r in records], dtype=float)
            cfd_raw_f = np.array([r["drag_force_b"] for r in records], dtype=float)
        else:
            print(f"[WARN] CFD drag log not found for test {test_id:04d}.")
        if cuboid_raw_t is not None:
            cuboid_mask = cuboid_raw_t <= plot_window
            cuboid_raw_t = cuboid_raw_t[cuboid_mask]
            cuboid_raw_f = cuboid_raw_f[cuboid_mask]
        if cfd_raw_t is not None:
            cfd_mask = cfd_raw_t <= plot_window
            cfd_raw_t = cfd_raw_t[cfd_mask]
            cfd_raw_f = cfd_raw_f[cfd_mask]
        fig_drag_components = plot_drag_components(
            f"{MODE_LABEL} Test{test_id:04d} Drag Logs (Trimmed)",
            cuboid_raw_t,
            cuboid_raw_f,
            cfd_raw_t,
            cfd_raw_f,
            t_vel,
            vel_frames["vel_b_plot"],
        )

        if args.save_dir:
            out_path = args.save_dir / f"{MODE_LABEL}_Test{test_id:04d}_vel_pwm.png"
            fig.savefig(out_path, dpi=150)
            out_traj = args.save_dir / f"{MODE_LABEL}_Test{test_id:04d}_traj_vectors.png"
            fig_traj.savefig(out_traj, dpi=150)
            if fig_speed_force is not None:
                out_speed = args.save_dir / f"{MODE_LABEL}_Test{test_id:04d}_speed_force_drag.png"
                fig_speed_force.savefig(out_speed, dpi=150)
            if fig_drag_highlight is not None:
                out_highlight = args.save_dir / f"{MODE_LABEL}_Test{test_id:04d}_thrust_drag_highlight.png"
                fig_drag_highlight.savefig(out_highlight, dpi=150)
            if fig_drag_components:
                for idx, fig_drag in enumerate(fig_drag_components):
                    suffix = ["Fx", "Fy", "Fz"][idx]
                    out_drag = args.save_dir / f"{MODE_LABEL}_Test{test_id:04d}_drag_{suffix}.png"
                    fig_drag.savefig(out_drag, dpi=150)

        if not show_plots:
            plt.close(fig)
            plt.close(fig_traj)
            if fig_speed_force is not None:
                plt.close(fig_speed_force)
            if fig_drag_highlight is not None:
                plt.close(fig_drag_highlight)
            for fig_drag in fig_drag_components:
                plt.close(fig_drag)

    if all_force_samples:
        coeffs = fit_front_rear_coeffs(all_force_samples)
        if coeffs is None:
            print("[WARN] Not enough samples to fit front/rear coefficients.")
        else:
            k_rear, k_front = coeffs
            mse = mse_for_coeffs(all_force_samples, coeffs)
            print(
                f"[INFO] Fitted coefficients: k_rear={k_rear:.4f}, k_front={k_front:.4f}"
            )
            if mse is not None:
                print(f"[INFO] Fit MSE: {mse:.4f}")

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
