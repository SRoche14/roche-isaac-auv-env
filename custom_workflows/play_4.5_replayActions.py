import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay PWM actions into Isaac Lab at sim step granularity.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import csv
import json

from rsl_rl.runners import OnPolicyRunner

import numpy as np
import math

import rosbag

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

# ---- user-editable inputs ----
# ACTIONS_BAG = "/home/warp/isaacsim4.5/IsaacLab/roche-isaac-auv-env/SimpleDragBag/Left/Left3_2025-11-19-15-45-44_pwm_command_list.bag"
# ACTIONS_BAG = "/home/warp/isaacsim4.5/IsaacLab/roche-isaac-auv-env/SimpleDragBag/Forward/Forward1_pwm_command_list_filtered_pwm_command_list.bag"
ACTIONS_BAG = "/home/warp/isaacsim4.5/IsaacLab/roche-isaac-auv-env/MITRE/Left/mitreLeft1_2025-12-16-14-09-31.bag"

ACTIONS_TOPIC = "/warpauv_1/control/motor_controller_feather/pwm_command_list"
# Match env ordering (thruster_dynamics.py): drive_left, drive_right, rear_left, rear_right, front_left, front_right
ACTION_ORDER = ["drive_left", "drive_right", "rear_left", "rear_right", "front_left", "front_right"]
ACTION_FIELD = "position"  # field within each motor command message

# Drag model selection: "mujoco_box", "steady_cfd", or "transient_cfd"
DRAG_MODEL = "cuboid"
DRAG_MODEL_LABELS = {
    "cuboid": "Cuboid",
    "steady_cfd": "CFD",
    "transient_cfd": "TransientCFD",
}

bag_number = 1

def pwm_to_action(x: np.ndarray) -> np.ndarray:
    """
    Optionally map raw PWM (or 'position') values from the bag into env action space.
    Currently identity; replace if your task expects normalized thruster inputs.
    """
    return x 


def load_bag_series(bag_path: str, topic: str):
    msgs = []
    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            msgs.append((t.to_sec(), msg))
    if not msgs:
        raise RuntimeError(f"No messages on {topic} in {bag_path}")
    return msgs


def message_to_action_vec(msg) -> np.ndarray:
    # Convert one ROS message -> numeric action vector in env order
    # Assumes msg.motor_commands is an iterable with .name and ACTION_FIELD attributes
    name_to_val = {m.name: getattr(m, ACTION_FIELD) for m in msg.motor_commands}
    return np.array([name_to_val.get(name, 0.0) for name in ACTION_ORDER], dtype=np.float32)


def compute_time_weighted_means(rel_times: np.ndarray, actions_np: np.ndarray, sim_dt: float) -> np.ndarray:
    """
    Compute time-weighted average action over each sim window [k*sim_dt, (k+1)*sim_dt).

    IMPORTANT FIX:
    If a window contains the final bag message and there is no later message,
    we treat that last segment as valid through the end of the window. This
    prevents the window from falling back to the previous mean.
    """
    assert rel_times.ndim == 1 and actions_np.ndim == 2
    assert len(rel_times) == actions_np.shape[0]
    action_dim = actions_np.shape[1]

    replay_end = float(rel_times[-1])
    n_windows = max(1, int(math.ceil(replay_end / sim_dt)))

    means = np.zeros((n_windows, action_dim), dtype=np.float32)
    last_mean = np.zeros((action_dim,), dtype=np.float32)

    i = 0  # index into message segments
    for k in range(n_windows):
        win_start = k * sim_dt
        win_end = win_start + sim_dt

        # advance to the first segment that might overlap this window
        while i + 1 < len(rel_times) and rel_times[i + 1] <= win_start:
            i += 1

        accum = np.zeros((action_dim,), dtype=np.float64)
        covered = 0.0

        seg_i = i
        seg_start = max(win_start, rel_times[seg_i])

        # ---- KEY CHANGE: last segment extends to this window's end ----
        seg_end_next = rel_times[seg_i + 1] if (seg_i + 1) < len(rel_times) else win_end
        seg_end = min(seg_end_next, win_end)

        while seg_start < win_end:
            dt = seg_end - seg_start
            if dt > 0.0:
                accum += actions_np[seg_i] * dt
                covered += dt

            if seg_end >= win_end:
                break  # window fully covered

            seg_i += 1
            if seg_i >= len(rel_times):
                # no further segments; nothing more to accumulate in this window
                break
            seg_start = max(rel_times[seg_i], win_start)
            seg_end_next = rel_times[seg_i + 1] if (seg_i + 1) < len(rel_times) else win_end
            seg_end = min(seg_end_next, win_end)

        if covered > 0.0:
            means[k] = (accum / covered).astype(np.float32)
            last_mean = means[k]
        else:
            # No overlap -> hold previous mean
            means[k] = last_mean
    
    return means


def _calculate_equivalent_box_dims(inertias: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    # ri = sqrt((3/(2m)) * (I_j + I_k - I_i))
    return torch.sqrt(
        (3.0 / (2.0 * masses.repeat(1, 3)))
        * (torch.roll(inertias, 1, 1) + torch.roll(inertias, -1, 1) - inertias)
    )


def mujoco_box_drag_forces(
    root_quat_w: torch.Tensor,
    root_linvel_w: torch.Tensor,
    root_angvel_w: torch.Tensor,
    inertias: torch.Tensor,
    masses: torch.Tensor,
    water_rho: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    root_quats_b = quat_conjugate(root_quat_w)
    root_linvels_b = quat_apply(root_quats_b, root_linvel_w)
    root_angvels_b = quat_apply(root_quats_b, root_angvel_w)

    ri = _calculate_equivalent_box_dims(inertias, masses)
    rj = torch.roll(ri, 1, 1)
    rk = torch.roll(ri, -1, 1)

    forces = -2.0 * water_rho * rj * rk * torch.abs(root_linvels_b) * root_linvels_b
    torques = (
        -0.5
        * water_rho
        * ri
        * (torch.pow(rj, 4) + torch.pow(rk, 4))
        * torch.abs(root_angvels_b)
        * root_angvels_b
    )
    return forces, torques


def main():
    """Play / replay using recorded PWM actions."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=1, use_fabric=not args_cli.disable_fabric  # single-env playback
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    # env.unwrapped.force_calculation_functions.use_transient_models = True

    # logging / checkpoint
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # If you want to force a particular checkpoint, keep your explicit override:
    resume_path = "/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-10-23_16-30-25/model_700.pt"

    # load previously trained model (not used for action selection, kept for completeness/export)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loaded model checkpoint from: {resume_path}")

    save_path = os.path.join(
        "source", "results", "rsl_rl", agent_cfg.experiment_name, agent_cfg.load_run, agent_cfg.load_checkpoint[:-3] + "_play"
    )
    os.makedirs(save_path, exist_ok=True)
    w = csv.writer(open(os.path.join(save_path, "output.csv"), "w"), delimiter=",")
    print(f"[INFO]: Saving results into: {save_path}")

    # optional export (disabled to reduce startup time during playback)
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_onnx(ppo_runner.alg.policy, path=export_model_dir, filename="policy.onnx")

    # --- read bag & build series ---
    msgs = load_bag_series(ACTIONS_BAG, ACTIONS_TOPIC)
    print(f"[INFO] Loaded {len(msgs)} PWM messages from {ACTIONS_BAG}")

    start_ts = msgs[0][0]
    print(start_ts)
    rel_times = np.array([ts - start_ts for (ts, _) in msgs], dtype=np.float64)

    actions_np = np.stack([message_to_action_vec(m) for (_, m) in msgs], axis=0)  # [N, 6]
    actions_np = pwm_to_action(actions_np)  # apply mapping if needed

    # sanity check on action dim
    env_act_dim = env.action_space.shape[-1]
    if env_act_dim != actions_np.shape[1]:
        raise ValueError(
            f"Action dim mismatch: bag has {actions_np.shape[1]} values per message, env expects {env_act_dim}."
        )

    # --- use one PWM command per sim step ---
    sim_dt = float(env.unwrapped.step_dt)
    expected_dt = 1.0 / 50.0
    if abs(sim_dt - expected_dt) > 1e-4:
        print(
            f"[WARN] step_dt={sim_dt:.6f}s differs from 1/50s; PWM replay may be time-misaligned."
        )
    actions_seq = actions_np
    print(f"[INFO] Using {len(actions_seq)} PWM commands at sim_dt={sim_dt:.6f}s (no averaging).")

    # --- run sim with precomputed actions ---
    # Preload transformer models/scalers if enabled to avoid first-step stall.
    base_env = env.unwrapped
    force_models = getattr(base_env, "force_calculation_functions", None)

    if DRAG_MODEL not in DRAG_MODEL_LABELS:
        raise ValueError(f"Invalid DRAG_MODEL '{DRAG_MODEL}'.")
    drag_label = DRAG_MODEL_LABELS[DRAG_MODEL]

    if force_models is not None:
        if DRAG_MODEL == "transient_cfd":
            force_models.use_transient_models = True
            print("[INFO] Preloading transient CFD transformer models...")
            force_models.debug = False
            force_models._ensure_transient_models_loaded()
        elif DRAG_MODEL == "steady_cfd":
            force_models.use_transient_models = False
        elif DRAG_MODEL == "mujoco_box":
            force_models.use_transient_models = False

    print("[INFO] Getting initial observations...")
    obs, _ = env.get_observations()
    device = env.unwrapped.device

    positions = []
    drag_log = []

    # Get your robot/vehicle articulation
    vehicle = getattr(base_env, "_robot")

    if vehicle is None:
        raise RuntimeError("Could not find vehicle articulation in the environment.")

    print("[INFO] Starting simulation replay...")
    warned_no_drag = False
    for k in range(len(actions_seq)):
        if not simulation_app.is_running():
            break
        a_step = actions_seq[k]  # np.float32 [6]
        print(a_step)
        a_step_t = torch.as_tensor(a_step, device=device, dtype=torch.float32)  # torch [6] on correct device
        # Start with all thrusters zeroed; uncomment individual lines to enable a thruster from the bagged action.
        action_tensor = torch.zeros((1, len(a_step)), device=device, dtype=torch.float32)  # [1,6]
        action_tensor[..., 0] = a_step_t[0]  # drive_left
        action_tensor[..., 1] = a_step_t[1]  # drive_right
        action_tensor[..., 2] = a_step_t[2]  # rear_left
        action_tensor[..., 3] = a_step_t[3]  # rear_right
        action_tensor[..., 4] = a_step_t[4]  # front_left
        action_tensor[..., 5] = a_step_t[5]  # front_right

        with torch.inference_mode():
            obs, rews, _, _ = env.step(action_tensor)
            root_state = vehicle.data.root_state_w   # tensor shape (num_robots, 13)
            pos = root_state[0, 0:3].detach().cpu().numpy().tolist()
            lin_vel_b = quat_apply(
                quat_conjugate(vehicle.data.root_quat_w),
                vehicle.data.root_lin_vel_w,
            )[0].detach().cpu().numpy().tolist()
            ang_vel_b = quat_apply(
                quat_conjugate(vehicle.data.root_quat_w),
                vehicle.data.root_ang_vel_w,
            )[0].detach().cpu().numpy().tolist()

            positions.append(pos)
            # simple logging as in your script
            distance = torch.norm(obs[0])  # FYI: this is the norm of the full obs vector
            w.writerow([rews[0].cpu().item(), distance.cpu().item()])

            if DRAG_MODEL == "mujoco_box":
                drag_f, drag_tau = mujoco_box_drag_forces(
                    vehicle.data.root_quat_w,
                    vehicle.data.root_lin_vel_w,
                    vehicle.data.root_ang_vel_w,
                    base_env.inertia_tensors,
                    base_env.masses,
                    base_env.cfg.water_rho,
                )
                drag_force = drag_f[0].detach().cpu().numpy().tolist()
                drag_torque = drag_tau[0].detach().cpu().numpy().tolist()
                drag_log.append(
                    {
                        "t": float(k * sim_dt),
                        "drag_force_b": drag_force,
                        "drag_torque_b": drag_torque,
                        "lin_vel_b": lin_vel_b,
                        "ang_vel_b": ang_vel_b,
                    }
                )
            elif force_models is not None:
                if hasattr(force_models, "predict_drag_components"):
                    f_lin, t_lin, f_ang, t_ang, _, _ = force_models.predict_drag_components(
                        vehicle.data.root_quat_w,
                        vehicle.data.root_lin_vel_w,
                        vehicle.data.root_ang_vel_w,
                    )
                    drag_force = (f_lin + f_ang)[0].detach().cpu().numpy().tolist()
                    drag_torque = (t_lin + t_ang)[0].detach().cpu().numpy().tolist()
                    drag_log.append(
                        {
                            "t": float(k * sim_dt),
                            "drag_force_b": drag_force,
                            "drag_torque_b": drag_torque,
                            "drag_force_linear_b": f_lin[0].detach().cpu().numpy().tolist(),
                            "drag_torque_linear_b": t_lin[0].detach().cpu().numpy().tolist(),
                            "drag_force_angular_b": f_ang[0].detach().cpu().numpy().tolist(),
                            "drag_torque_angular_b": t_ang[0].detach().cpu().numpy().tolist(),
                            "lin_vel_b": lin_vel_b,
                            "ang_vel_b": ang_vel_b,
                        }
                    )
                else:
                    f_d, g_d, f_v, g_v = force_models.calculate_density_and_viscosity_forces(
                        vehicle.data.root_quat_w,
                        vehicle.data.root_lin_vel_w,
                        vehicle.data.root_ang_vel_w,
                        base_env.inertia_tensors,
                        base_env.inertia_tensors_mean,
                        base_env.cfg.water_beta,
                        base_env.cfg.water_rho,
                        base_env.masses,
                    )
                    drag_force = (f_d + f_v)[0].detach().cpu().numpy().tolist()
                    drag_torque = (g_d + g_v)[0].detach().cpu().numpy().tolist()
                    drag_log.append(
                        {
                            "t": float(k * sim_dt),
                            "drag_force_b": drag_force,
                            "drag_torque_b": drag_torque,
                            "lin_vel_b": lin_vel_b,
                            "ang_vel_b": ang_vel_b,
                        }
                    )
            elif not warned_no_drag:
                print("[WARN] Drag model not available; skipping drag logging.")
                warned_no_drag = True

    # close the simulator
    env.close()

    folder_name = os.path.basename(os.path.dirname(ACTIONS_BAG))
    positions_path = os.path.join(save_path, f"positionsMITRE{folder_name}{drag_label}{bag_number}.csv")
    with open(positions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        writer.writerows(positions)
    print(f"[INFO] Saved all positions to {positions_path}")

    drag_path = os.path.join(save_path, f"drag_forces_moments_{drag_label}{bag_number}.json")
    with open(drag_path, "w") as f:
        json.dump(
            {
                "drag_model": DRAG_MODEL,
                "records": drag_log,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Saved drag forces/torques to {drag_path}")

if __name__ == "__main__":
    main()
    simulation_app.close()
