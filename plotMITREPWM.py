#!/usr/bin/env python3
"""
Plot PWM command time series for each MITRE rosbag.

Each bag produces a plot with 6 thruster lines (when available) and time on the x-axis.
Titles include the trajectory type and associated Curee test number.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rosbag
from genpy.dynamic import generate_dynamic

PWM_TOPIC = "/warpauv_1/control/motor_controller_feather/pwm_command_list"
DEADBAND = 0.08
DEFAULT_ROTOR_CONSTANT = 0.0002
DEFAULT_DYN_TIME_CONSTANT = 0.05
FORWARD_SCALE = 0.8
FORWARD_A = -139.0
FORWARD_B = 500.0
FORWARD_C = 8.28
BACK_A = 161.0
BACK_B = 517.86
BACK_C = -5.72
FORWARD_BIAS = 1.0
BACK_BIAS = 0.5
THRUSTER_ORDER = (
    "drive_left",
    "drive_right",
    "rear_right",
    "front_right",
    "front_left",
    "rear_left",
)

DATASETS = [
    {
        "label": "Forward",
        "base_dir": Path("MITRE/Forward"),
        "bag_prefix": "mitreForward",
        "excluded_tests": {7, 12},
        "mapping": None,
    },
    {
        "label": "Left",
        "base_dir": Path("MITRE/Left"),
        "bag_prefix": "mitreLeft",
        "excluded_tests": set(),
        "mapping": {1: 6, 3: 8, 4: 9, 5: 10},
    },
    {
        "label": "Up",
        "base_dir": Path("MITRE/Up"),
        "bag_prefix": "mitreUp",
        "excluded_tests": {7, 12},
        "mapping": None,
    },
    {
        "label": "LTurn",
        "base_dir": Path("MITRE/LTurn"),
        "bag_prefix": "mitreLTurn",
        "excluded_tests": {7, 12, 17, 21},
        "mapping": None,
    },
    {
        "label": "Circles",
        "base_dir": Path("MITRE/Circles"),
        "bag_prefix": "mitreCircle",
        "excluded_tests": {7, 12},
        "mapping": None,
    },
]


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


def build_run_mapping(base_dir: Path, bag_prefix: str, excluded_tests: set) -> Dict[int, int]:
    tests = discover_test_ids(base_dir)
    runs = discover_run_ids_from_bags(base_dir, bag_prefix)
    if len(runs) != len(tests):
        raise ValueError(
            f"Run/test count mismatch in {base_dir}: {len(runs)} runs vs {len(tests)} tests."
        )
    mapping = {run_id: test_id for run_id, test_id in zip(runs, tests)}
    return {
        run_id: test_id for run_id, test_id in mapping.items() if test_id not in excluded_tests
    }


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


def plot_pwm(
    times: np.ndarray,
    names: List[str],
    values: np.ndarray,
    title: str,
    ylabel: str = "PWM command",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(names):
        ax.plot(times, values[:, idx], label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_pwm_separate(
    times: np.ndarray,
    names: List[str],
    values: np.ndarray,
    title: str,
    ylabel: str,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        if idx >= len(names):
            ax.axis("off")
            continue
        ax.plot(times, values[:, idx], color="tab:blue")
        ax.set_title(names[idx])
        ax.grid(True)
        if idx % 2 == 0:
            ax.set_ylabel(ylabel)
    for ax in axes[-2:]:
        ax.set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


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


def apply_thruster_dynamics(
    motor_values: np.ndarray, times: np.ndarray, tau: float
) -> np.ndarray:
    if len(motor_values) == 0:
        return motor_values
    if len(times) != len(motor_values):
        raise ValueError("times and motor_values must have the same length")
    # Matches thruster_dynamics.DynamicsFirstOrder.update (alpha forced to zero).
    return np.array(motor_values, copy=True)


def convert_motor_values_to_force(
    motor_values: np.ndarray, rotor_constant: float
) -> np.ndarray:
    # ConversionFunctionBasic in thruster_dynamics.py: rotorConstant * abs(cmd) * cmd
    return rotor_constant * np.abs(motor_values) * motor_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PWM commands for all MITRE rosbags.")
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
        "--topic",
        type=str,
        default=PWM_TOPIC,
        help=f"PWM command topic (default: {PWM_TOPIC})",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Curee test id (1-23) to plot as six separate thruster panels.",
    )
    parser.add_argument(
        "--rotor-constant",
        type=float,
        default=DEFAULT_ROTOR_CONSTANT,
        help="Rotor constant used for force conversion (default matches warpauv_env).",
    )
    parser.add_argument(
        "--dyn-time-constant",
        type=float,
        default=DEFAULT_DYN_TIME_CONSTANT,
        help="Thruster dynamics time constant (default matches warpauv_env).",
    )
    args = parser.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    show_plots = not args.no_show

    target_test_id = args.run_id
    if target_test_id is not None and not (1 <= target_test_id <= 23):
        raise SystemExit("--run-id must be between 1 and 23")

    found_target = False

    for dataset in DATASETS:
        base_dir = dataset["base_dir"]
        bag_prefix = dataset["bag_prefix"]
        if not base_dir.exists():
            print(f"[WARN] Missing directory: {base_dir}")
            continue

        if dataset["mapping"] is None:
            try:
                mapping = build_run_mapping(base_dir, bag_prefix, dataset["excluded_tests"])
            except ValueError as exc:
                print(f"[WARN] {dataset['label']}: {exc}")
                continue
        else:
            mapping = dataset["mapping"]

        bag_paths = sorted(base_dir.glob(f"{bag_prefix}*.bag"))
        for bag_path in bag_paths:
            match = re.search(rf"{bag_prefix}(\d+)", bag_path.name)
            if not match:
                print(f"[WARN] Unable to parse run id from {bag_path.name}")
                continue
            run_id = int(match.group(1))
            if run_id not in mapping:
                print(f"[WARN] Skipping {bag_path.name}: no mapped test id for run {run_id}")
                continue

            test_id = mapping[run_id]
            if target_test_id is not None and test_id != target_test_id:
                continue
            if target_test_id is not None:
                found_target = True
            title = f"{dataset['label']} Test{test_id:04d} PWM"

            try:
                times, names, values = load_pwm_series(bag_path, args.topic, THRUSTER_ORDER)
            except Exception as exc:
                print(f"[WARN] {bag_path.name}: {exc}")
                continue

            motor_values = pwm_to_motor_values(values)
            motor_values = apply_thruster_dynamics(
                motor_values, times, args.dyn_time_constant
            )
            force_values = convert_motor_values_to_force(
                motor_values, args.rotor_constant
            )

            if target_test_id is None:
                fig_pwm = plot_pwm(times, names, values, title, ylabel="PWM command")
                fig_force = plot_pwm(
                    times,
                    names,
                    force_values,
                    title.replace("PWM", "Force"),
                    ylabel="Estimated thrust (N)",
                )
            else:
                fig_pwm = plot_pwm_separate(
                    times, names, values, title, ylabel="PWM command"
                )
                fig_force = plot_pwm_separate(
                    times,
                    names,
                    force_values,
                    title.replace("PWM", "Force"),
                    ylabel="Estimated thrust (N)",
                )

            if args.save_dir:
                safe_label = dataset["label"].replace(" ", "")
                suffix_pwm = "pwm" if target_test_id is None else "pwm_thrusters"
                out_pwm = args.save_dir / f"{safe_label}_Test{test_id:04d}_{suffix_pwm}.png"
                fig_pwm.savefig(out_pwm, dpi=150)

                suffix_force = "force" if target_test_id is None else "force_thrusters"
                out_force = args.save_dir / f"{safe_label}_Test{test_id:04d}_{suffix_force}.png"
                fig_force.savefig(out_force, dpi=150)

            if not show_plots:
                plt.close(fig_pwm)
                plt.close(fig_force)

    if show_plots:
        plt.show()

    if target_test_id is not None and not found_target:
        print(f"[WARN] No bag found for test id {target_test_id:04d}.")


if __name__ == "__main__":
    main()
