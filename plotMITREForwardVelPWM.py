#!/usr/bin/env python3
"""
Plot MITRE Forward velocity (x,y,z) alongside thruster motor rad/s per test.

For each Curee Test JSON in MITRE/Forward, generate a 3x2 subplot figure.
Each subplot overlays the same x/y/z velocity with one thruster's motor rad/s.
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

BASE_DIR = Path("MITRE/Forward")
START_TIMES_JSON = BASE_DIR / "startingForwardTimes.json"
BAG_PREFIX = "mitreForward"

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

# Keep the same order used in plotMITREPWM.py for consistent labeling.
THRUSTER_ORDER = (
    "drive_left",
    "drive_right",
    "rear_right",
    "front_right",
    "front_left",
    "rear_left",
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
    if traj.shape[0] < 2:
        return np.empty((0, 3), dtype=float)
    return np.diff(traj, axis=0) * float(freq_hz)


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


def load_start_times(prefix: str) -> Dict[int, float]:
    if not START_TIMES_JSON.exists():
        return {}
    with START_TIMES_JSON.open("r") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith(prefix):
            run_id = int(k.replace(prefix, ""))
            out[run_id] = float(v)
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
    t_pwm: np.ndarray,
    motor_values: np.ndarray,
    pwm_names: List[str],
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    axes = axes.ravel()

    name_to_idx = {name: idx for idx, name in enumerate(pwm_names)}
    handles = None
    labels = None

    for ax, thruster in zip(axes, THRUSTER_ORDER):
        ax.plot(t_vel, velocity[:, 0], label="vx", color="tab:blue", linewidth=1.0)
        ax.plot(t_vel, velocity[:, 1], label="vy", color="tab:orange", linewidth=1.0)
        ax.plot(t_vel, velocity[:, 2], label="vz", color="tab:green", linewidth=1.0)
        ax.set_title(thruster)
        ax.set_ylabel("Velocity [m/s]")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        if thruster in name_to_idx:
            idx = name_to_idx[thruster]
            line = ax2.plot(
                t_pwm,
                motor_values[:, idx],
                label="motor rad/s",
                color="tab:red",
                linewidth=1.0,
                alpha=0.8,
            )
            if handles is None:
                handles = ax.get_lines()[:3] + line
                labels = [h.get_label() for h in handles]
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

    for ax in axes[-2:]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(f"Forward Test{test_id:04d} Velocity vs Motor rad/s")
    if handles and labels:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Forward Curee test velocity with thruster motor rad/s."
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
        "--run-id",
        type=int,
        default=None,
        help="Curee test id to plot (default: all tests in MITRE/Forward).",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=PWM_TOPIC,
        help=f"PWM command topic (default: {PWM_TOPIC})",
    )
    args = parser.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    qualy_starts = load_start_times(prefix="qualyForward")
    mitre_starts = load_start_times(prefix="mitreForward")

    tests = collect_tests(BASE_DIR, args.run_id)
    if not tests:
        print("[WARN] No Curee Test JSON files found.")
        return

    try:
        test_to_bag = build_test_to_bag(BASE_DIR, BAG_PREFIX)
    except ValueError as exc:
        print(f"[WARN] {exc}")
        return

    show_plots = not args.no_show

    for test_id, gt_path in tests:
        try:
            source_type, source_name, positions, freq_hz = load_positions(gt_path)
        except Exception as exc:
            print(f"[WARN] Skipping {gt_path}: {exc}")
            continue

        start_gt = qualy_starts.get(test_id, 0.0)
        if test_id not in qualy_starts:
            print(f"[WARN] Missing qualyForward start for test {test_id}; using 0.0s")

        bag_path = test_to_bag.get(test_id)
        if bag_path is None:
            print(f"[WARN] No rosbag found for test {test_id:04d}; skipping PWM overlay.")
            continue

        try:
            t_pwm, pwm_names, pwm_values = load_pwm_series(bag_path, args.topic, THRUSTER_ORDER)
        except Exception as exc:
            print(f"[WARN] {bag_path.name}: {exc}")
            continue

        start_mitre = mitre_starts.get(test_id, 0.0)
        if test_id not in mitre_starts:
            print(f"[WARN] Missing mitreForward start for test {test_id}; using 0.0s")

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
        positions_smooth = spike_smoothing(positions_trim, freq_hz=freq_hz)
        velocity = compute_velocity(positions_smooth, freq_hz)
        if velocity.size == 0:
            print(f"[WARN] {gt_path} has fewer than 2 samples after trim; skipping.")
            continue
        t_vel = np.arange(velocity.shape[0]) / float(freq_hz)

        t_pwm, pwm_values = trim_time_series(t_pwm, pwm_values, start_mitre)
        t_pwm, pwm_values = limit_time_series(t_pwm, pwm_values, plot_window)

        if len(t_pwm) == 0:
            print(f"[WARN] No PWM data after trimming for test {test_id:04d}; skipping.")
            continue

        motor_values = pwm_to_motor_values(pwm_values)

        print(
            f"[INFO] Test{test_id:04d}: {source_type} '{source_name}', "
            f"{len(positions)} samples @ {freq_hz:.1f} Hz"
        )
        fig = plot_velocity_pwm(test_id, t_vel, velocity, t_pwm, motor_values, pwm_names)

        if args.save_dir:
            out_path = args.save_dir / f"Forward_Test{test_id:04d}_vel_pwm.png"
            fig.savefig(out_path, dpi=150)
        if not show_plots:
            plt.close(fig)

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
