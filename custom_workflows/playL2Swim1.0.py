import argparse
import os
import torch
import gc
import csv
import gymnasium as gym
import matplotlib.pyplot as plt
import json, numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a trained RSL-RL agent vectorized over many envs.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

RUN_DIR = os.path.join("results", "play_runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)
JSONL_PATH = os.path.join(RUN_DIR, "results.jsonl")
CSV_PATH = os.path.join(RUN_DIR, "summary.csv")

def _write_csv_row(path, row, fieldnames):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

def _compute_boxplot_stats(values, whis=1.5):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None,
            "iqr": None, "whis_low": None, "whis_high": None, "outliers": [],
            "mean": None, "std": None
        }
    arr = np.sort(arr)
    q1 = float(np.percentile(arr, 25))
    median = float(np.percentile(arr, 50))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lowb = q1 - 1.5 * iqr
    highb = q3 + 1.5 * iqr
    whis_low = float(arr[arr >= lowb].min())
    whis_high = float(arr[arr <= highb].max())
    outliers = arr[(arr < whis_low) | (arr > whis_high)].tolist()
    return {
        "n": int(arr.size),
        "min": float(arr[0]),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(arr[-1]),
        "iqr": float(iqr),
        "whis_low": whis_low,
        "whis_high": whis_high,
        "outliers": outliers,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }

def append_results(*, goal_o, starting_o, vel, offset, num_envs, task, checkpoint,
                   eval_dt, timeout_s, success_radius_m, successes, failures,
                   avg_energy, success_times):
    success_times = list(success_times)
    box = _compute_boxplot_stats(success_times)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "checkpoint": checkpoint,
        "params": {
            "goal_o": goal_o, "starting_o": starting_o, "vel": vel, "offset": offset,
            "num_envs": num_envs, "eval_dt": eval_dt, "timeout_s": timeout_s,
            "success_radius_m": success_radius_m,
        },
        "metrics": {
            "successes": successes,
            "failures": failures,
            "success_rate": successes / max(1, successes + failures),
            "avg_energy": avg_energy,
        },
        "stop_boxplot": box,          # <-- full boxplot stats here
        "success_times": success_times,
    }

    # JSONL: full detail (incl. outliers list)
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    # CSV: tidy scalar summary
    row = {
        "timestamp": record["timestamp"],
        "task": task,
        "checkpoint": checkpoint,
        "goal_o": goal_o, "starting_o": starting_o, "vel": vel, "offset": offset,
        "num_envs": num_envs, "eval_dt": eval_dt, "timeout_s": timeout_s,
        "success_radius_m": success_radius_m,
        "successes": successes, "failures": failures,
        "success_rate": record["metrics"]["success_rate"],
        "avg_energy": avg_energy,
        "stop_n": box["n"], "stop_min": box["min"], "stop_q1": box["q1"],
        "stop_median": box["median"], "stop_q3": box["q3"], "stop_max": box["max"],
        "stop_iqr": box["iqr"], "stop_whis_low": box["whis_low"],
        "stop_whis_high": box["whis_high"], "stop_mean": box["mean"],
        "stop_std": box["std"], "stop_outliers_count": len(box["outliers"]),
    }
    _write_csv_row(CSV_PATH, row, list(row.keys()))

def main():
    """Play with RSL-RL agent across many environments."""

    goal_orientations = [[1.0, 0.0, 0.0,0.0], [0.70710678,  0.0, 0.0,  0.70710678], [0.0, 0.0, 0.0, 1.0], [0.70710678, 0.0, 0.0, -0.70710678]]
    starting_orientations = [[1.0, 0.0, 0.0,0.0], [0.70710678,  0.0, 0.0,  0.70710678], [0.0, 0.0, 0.0, 1.0], [0.70710678, 0.0, 0.0, -0.70710678]] 
    # init_velocities = [[0,0,0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    offsets = [[0.5, 0, 0], [2.5, 0, 0], [5.5, 0, 0]]

    init_velocities = [[0,0,0]]

    for goal_o in goal_orientations:
        for starting_o in starting_orientations:
            for vel in init_velocities[:1]:
                for offset in offsets:
                    num_envs = args_cli.num_envs

                    # Build env cfg; we do NOT force CPU here; Fabric toggle is respected
                    env_cfg = parse_env_cfg(
                        args_cli.task,
                        num_envs=num_envs,
                        use_fabric=not args_cli.disable_fabric,
                    )
                    env_cfg.cap_episode_length = False
                    env_cfg.goal_o = goal_o
                    env_cfg.starting_o = starting_o
                    env_cfg.vel = vel
                    env_cfg.offset = offset
                    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

                    # Create environment and wrap for RSL-RL
                    env = gym.make(args_cli.task, cfg=env_cfg)
                    env = RslRlVecEnvWrapper(env)

                    # --------------------
                    # Load trained policy
                    # --------------------
                    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
                    log_root_path = os.path.abspath(log_root_path)
                    print(f"[INFO] Loading experiment from directory: {log_root_path}")
                    # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                    # print(f"[INFO]: Loading model checkpoint from: {resume_path}")

                    # If you want to hard-pin a checkpoint, uncomment/replace:
                    # for cfd with random vels
                    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-26-10/model_2750.pt'

                    # for cfd with no random vels
                    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-34-03/model_2450.pt'

                    # cfd with no random vels and energy saving
                    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-38-40/model_3200.pt'

                    # cfd with random vels and energy saving
                    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-43-38/model_2800.pt'

                    # for rec. with random vels
                    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-47-16/model_1900.pt'

                    # for rec. with no random vels
                    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-50-35/model_1600.pt'

                    # for rec. with no random vels and energy saving
                    # resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-53-53/model_2150.pt'

                    # for rec. with random vels and energy saving
                    resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-11_14-57-13/model_2150.pt'


                    ########################################
                    # Asymmetric models
                    ########################################

                    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                    ppo_runner.load(resume_path)
                    print(f"[INFO]: Loaded model checkpoint from: {resume_path}")

                    # Optionally export to ONNX (kept from your script)
                    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
                    export_policy_as_onnx(ppo_runner.alg.policy, path=export_model_dir, filename="policy.onnx")

                    # --------------------
                    # Inference setup
                    # --------------------
                    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

                    # independent of the simulator's internal dt.
                    eval_dt = (1/120)
                    if offset[0] == 0.5:
                        timeout_s = 2.0
                    elif offset[0] == 2.5:
                        timeout_s = 4.0
                    elif offset[0] == 5.5:
                        timeout_s = 7.0
                    else:
                        raise Exception("Something is wrong with the offsets!")
                    success_radius_m = 0.1

                    # Track per-env state
                    # times_to_stop[i]: time when env i first reaches success_radius; stays -1 for failures
                    times_to_stop = torch.full((num_envs,), -1.0, device=env.unwrapped.device)
                    # Elapsed time per env (for failure cutoff)
                    elapsed = torch.zeros((num_envs,), device=env.unwrapped.device)
                    # Active mask: True if env still running (neither succeeded nor failed)
                    active = torch.ones((num_envs,), dtype=torch.bool, device=env.unwrapped.device)
                    # energy tensor for each environment
                    total_energy = torch.zeros((num_envs,), device=env.unwrapped.device)

                    # Optional: a CSV for debugging (commented out by default)
                    # save_path = os.path.join("source", "results", "rsl_rl", agent_cfg.experiment_name, agent_cfg.load_run, (agent_cfg.load_checkpoint or "ckpt").replace(".pt", "_play"))
                    # os.makedirs(save_path, exist_ok=True)
                    # w = csv.writer(open(os.path.join(save_path, "output.csv"), 'w'), delimiter=',')
                    # print(f"[INFO]: Saving step data into: {save_path}")

                    # --------------------
                    # Reset and simulate
                    # --------------------
                    obs, _ = env.get_observations()

                    # Sanity: obs layout from your env is:
                    # [goal_quat(4), offset_from_origin_b(3), root_quat_w(4), root_lin_vel_b(3), root_ang_vel_b(3)] => 17 dims
                    OFFSET_START = 4
                    OFFSET_END = 7  # exclusive

                    # Run until all envs are resolved or app is closed
                    # Vectorized loop; NO per-env loop
                    while simulation_app.is_running() and active.any():
                        with torch.inference_mode():
                            actions = policy(obs)
                            total_energy += torch.sum(actions**2, dim=1) * eval_dt
                            obs, rews, _, _ = env.step(actions)

                            # Distance to target (origin) from body-frame offset in obs
                            offsets_b = obs[:, OFFSET_START:OFFSET_END]
                            dists = torch.norm(offsets_b, dim=1)
                            
                            lin_speeds = torch.norm(obs[:, 11:14], dim=1)

                            # Update elapsed time for active envs
                            elapsed = elapsed + (eval_dt * active.float())

                            # Success = close enough *and* slow enough
                            newly_success = (dists <= success_radius_m) & (lin_speeds <= 0.5) & active
                            times_to_stop[newly_success] = elapsed[newly_success]
                            active[newly_success] = False

                            # Check timeouts (failures) for still-active envs
                            newly_failed = (elapsed >= timeout_s) & active
                            active[newly_failed] = False

                            # Optional: write per-step aggregates
                            # mean_rew = rews.mean().item()
                            # mean_dist = dists.mean().item()
                            # w.writerow([mean_rew, mean_dist])

                    # --------------------
                    # Summaries
                    # --------------------
                    avg_energy = total_energy.mean().item()
                    print(f"\nAverage energy usage across all environments: {avg_energy:.4f} (arb. units)")

                    successes = (times_to_stop >= 0).sum().item()
                    failures = num_envs - successes

                    # Gather success times on CPU
                    success_times = times_to_stop[times_to_stop >= 0].detach().cpu().tolist()

                    print("\n==== Evaluation Summary ====")
                    print(f"Num envs:        {num_envs}")
                    print(f"Success radius:  {success_radius_m} m")
                    print(f"Timeout:         {timeout_s} s")
                    print(f"Eval dt:         {eval_dt} s")
                    print(f"Successes:       {successes}")
                    print(f"Failures:        {failures}")
                    print(f"Times to stop (s) for successes ({len(success_times)}):")
                    print(success_times)

                    append_results(
                        goal_o=goal_o, starting_o=starting_o, vel=vel, offset=offset,
                        num_envs=num_envs, task=args_cli.task, checkpoint=resume_path,
                        eval_dt=eval_dt, timeout_s=timeout_s, success_radius_m=success_radius_m,
                        successes=successes, failures=failures, avg_energy=avg_energy,
                        success_times=success_times,
                    )

                    env.close()
                    del env
                    del ppo_runner
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
