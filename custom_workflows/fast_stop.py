#!/usr/bin/env python3
"""
Play: launch toward target and stop using a trained RL policy (RSL-RL).

- Spawns the vehicle close to the target (origin) at a configurable offset.
- Gives it an initial 10 m/s velocity toward the target.
- Runs the trained policy to control the vehicle.
- Detects when the vehicle has come to rest at the target and exits.
- Logs telemetry to CSV.

Requires:
- Your WarpAUV IsaacLab task (default task name below; override with --task).
- The specified RSL-RL checkpoint.

"""

import argparse
import time

from isaaclab.app import AppLauncher

# Local imports
import cli_args  # uses your existing RSL-RL CLI schema

# RSL-RL runner + IsaacLab helpers
import gymnasium as gym
import os
import torch
import csv
import isaaclab.utils.math as math_utils

parser = argparse.ArgumentParser(description="Play with a trained RSL-RL policy: stop at a nearby target.")
# Scenario
parser.add_argument("--task", type=str, default="WarpAUV-Direct-v0", help="Registered IsaacLab task.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (use 1 for this demo).")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--init_offset_m", type=float, default=0.5, help="Initial distance from target (m).")
parser.add_argument("--approach_speed", type=float, default=5.0, help="Initial speed toward target (m/s).")
parser.add_argument("--log_dir", type=str, default="logs/stop_with_policy", help="CSV log directory.")

# Stop detection thresholds
parser.add_argument("--pos_eps", type=float, default=0.2, help="Stop threshold on position error (m).")
parser.add_argument("--vel_eps", type=float, default=0.05, help="Stop threshold on speed (m/s).")
parser.add_argument("--hold_steps", type=int, default=10, help="Consecutive frames to hold stop before exit.")

# Isaac app / fabric flags
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric / use USD I/O.")

# RSL-RL CLI (keeps compatibility with your config/alg args, though we mainly just load a checkpoint here)
cli_args.add_rsl_rl_args(parser)
# App launcher args (e.g., --headless)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Seed for reproducibility
torch.manual_seed(args_cli.seed)

# Launch Omniverse App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def initialize_close_and_fast(isaac_env, offset_m=2.0, approach_speed=10.0):
    """
    Place the robot near the target (environment origin) and give it an initial velocity toward the origin.
    - Spawn at +X relative to origin, same depth as default.
    - Give world-frame velocity of -approach_speed along X (toward origin).
    """
    device = isaac_env.device
    env_ids = isaac_env._robot._ALL_INDICES
    N = isaac_env.num_envs

    default_state = isaac_env._robot.data.default_root_state[env_ids].clone()  # (N,13)
    origins = isaac_env.scene.env_origins[env_ids]  # (N,3)

    pose_w = torch.zeros((N, 7), device=device)
    # Position: (origin_x + offset, origin_y, default_depth)
    pose_w[:, :3] = torch.stack(
        [origins[:, 0] + offset_m, origins[:, 1], default_state[:, 2]], dim=-1
    )
    # Orientation: identity quaternion
    pose_w[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(N, 1)

    # Linear velocity toward origin in world frame (-X); angular vel zeros
    vel_w = torch.zeros((N, 6), device=device)
    vel_w[:, 0] = -abs(approach_speed)
    # vel_w[:, 1] = math_utils.quat_apply(isaac_env._robot.root_link_quat_w, isaac_env._robot.root_lin_vel_b)

    # Write to simulator
    isaac_env._robot.write_root_pose_to_sim(pose_w, env_ids)
    isaac_env._robot.write_root_velocity_to_sim(vel_w, env_ids)

    # Keep env mirrors in sync (useful for observations/rewards that reference these)
    isaac_env._default_root_state[env_ids, :7] = pose_w
    isaac_env._default_root_state[env_ids, 7:] = vel_w
    isaac_env._default_env_origins[env_ids, :] = torch.stack(
        [origins[:, 0], origins[:, 1], default_state[:, 2]], dim=-1
    )


def main():

    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Fixed checkpoint path
    # asymmetric cfd housing path
    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-13_22-22-34/model_500.pt'
    # asym with fast stop training:
    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-14_17-56-38/model_750.pt'
    # regular cfd path
    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-26-10/model_2750.pt'

    ### do not use these
    # cuboid path
    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-47-16/model_1900.pt'
    # cuboid with fast stop training
    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-14_18-13-50/model_2050.pt'
    ###

    # airplane with fast stop training
    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-14_21-54-34/model_600.pt'


    # asym cfd with fast stop training (5 m/s) and no DR
    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-15_21-38-51/model_450.pt'

    # cuboid with fast stop training (5 m/s) and no DR
    resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-15_21-28-10/model_350.pt'

    # Force deterministic setup (no domain rand / random spawns) if your cfg supports it
    if hasattr(env_cfg, "eval_mode"):
        env_cfg.eval_mode = True
    if hasattr(env_cfg, "scene"):
        env_cfg.scene.num_envs = args_cli.num_envs

    isaac_env = env.unwrapped

    # === Reset and set starting conditions ===
    isaac_env._reset_idx(isaac_env._robot._ALL_INDICES)
    initialize_close_and_fast(
        isaac_env,
        offset_m=args_cli.init_offset_m,
        approach_speed=args_cli.approach_speed,
    )

    # === Prepare RSL-RL policy runner and load checkpoint ===
    # Parse agent cfg using your existing CLI (keeps device/normalization settings consistent)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_dir = None  # no new training logs in "play"
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    print("[INFO] Loading policy:", resume_path)
    ppo_runner.load(resume_path)

    # Inference policy (handles obs normalization inside)
    policy = ppo_runner.get_inference_policy(device=isaac_env.device)

    # (Optional) export for inspection
    # export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # from isaaclab_rl.rsl_rl import export_policy_as_onnx
    # export_policy_as_onnx(ppo_runner.alg.policy, path=export_dir, filename="policy.onnx")

    # Prime observations
    obs, _ = env.get_observations()

    # === CSV logger ===
    # os.makedirs(args.log_dir, exist_ok=True)
    # csv_path = os.path.join(args.log_dir, "telemetry.csv")
    # f = open(csv_path, "w")
    # f.write("t,pos_err_x,pos_err_y,pos_err_z,vel_x,vel_y,vel_z,reward,stopped\n")
    # f.flush()
    # print(f"[INFO] Logging to: {os.path.abspath(csv_path)}")

    # Stop detection thresholds
    POS_EPS = args_cli.pos_eps
    VEL_EPS = args_cli.vel_eps
    HOLD_STEPS = args_cli.hold_steps
    hold_counter = 0
    t0 = time.time()

    # === Control loop ===
    while simulation_app.is_running():
        with torch.inference_mode():
            # Policy action
            actions = policy(obs)  # expects torch tensor, returns (N, action_dim)

            # Step the environment
            obs, rews, terminated, truncated = env.step(actions)

            # Extract body-frame position error and velocity from obs
            # Env obs layout (per your WarpAUV env):
            # [0:4] goal quaternion
            # [4:7] offset_from_origin_b (m)  <- position error in body frame
            # [7:11] root quaternion (wxyz)
            # [11:14] root_lin_vel_b (m/s)
            # [14:17] root_ang_vel_b (rad/s)
            pos_b = obs[0, 4:7]
            vel_b = obs[0, 11:14]
            print(obs[:, 4:7])
            # temporarily turn off trying to hit goal orientation
            obs[:, 0:4] = obs[:, 7:11]


            # Stop condition: close in position and nearly zero speed
            pos_close = torch.linalg.norm(pos_b).item() <= POS_EPS
            vel_small = torch.linalg.norm(vel_b).item() <= VEL_EPS
            stopped = pos_close 
            hold_counter = hold_counter + 1 if stopped else 0

            # Log telemetry
            t = time.time() - t0

            # # If held stop long enough, exit
            if hold_counter >= HOLD_STEPS:
                print("[INFO] Target reached and vehicle stopped under policy control. Exiting.")
                print("time: ", t)
                break

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
