import argparse
import os
import torch
import gc
import csv
import gymnasium as gym
import json, numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a trained RSL-RL agent vectorized over many envs (single-sim, fast, no inference_mode).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
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
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import (
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
        "stop_boxplot": box,
        "success_times": success_times,
    }

    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

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

def _cfg_obj(env):
    if hasattr(env.unwrapped, "task") and hasattr(env.unwrapped.task, "cfg"):
        return env.unwrapped.task.cfg
    if hasattr(env.unwrapped, "cfg"):
        return env.unwrapped.cfg
    return None

def _set_trial_cfg(env, *, goal_o, starting_o, vel, offset):
    cfg = _cfg_obj(env)
    if cfg is None:
        return False
    # flat lists
    goal_o = np.asarray(goal_o, dtype=float).tolist()
    starting_o = np.asarray(starting_o, dtype=float).tolist()
    vel = np.asarray(vel, dtype=float).tolist()
    offset = np.asarray(offset, dtype=float).tolist()
    if hasattr(cfg, "goal_o"):      cfg.goal_o = goal_o
    if hasattr(cfg, "starting_o"):  cfg.starting_o = starting_o
    if hasattr(cfg, "vel"):         cfg.vel = vel
    if hasattr(cfg, "offset"):      cfg.offset = offset
    return True

def main():
    # DO NOT use torch.inference_mode globally; no_grad is enough and safe
    torch.set_grad_enabled(False)

    goal_orientations = [
        [1.0, 0.0, 0.0, 0.0],
        [0.70710678, 0.0, 0.0, 0.70710678],
        [0.0, 0.0, 0.0, 1.0],
        [0.70710678, 0.0, 0.0, -0.70710678],
    ]
    starting_orientations = [
        [1.0, 0.0, 0.0, 0.0],
        [0.70710678, 0.0, 0.0, 0.70710678],
        [0.0, 0.0, 0.0, 1.0],
        [0.70710678, 0.0, 0.0, -0.70710678],
    ]
    offsets = [[0.5, 0, 0], [2.5, 0, 0], [5.5, 0, 0]]
    init_velocities = [[10,0,0], [-10,0,0], [0,10,0], [0,-10,0], [0,0,10], [0,0,-10]]
    num_envs = args_cli.num_envs

    # Build env ONCE; seed cfg so initial internal reset is valid
    base_cfg = parse_env_cfg(
        args_cli.task,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    base_cfg.cap_episode_length = False
    base_cfg.goal_o = goal_orientations[0]
    base_cfg.starting_o = starting_orientations[0]
    base_cfg.vel = init_velocities[0]
    base_cfg.offset = offsets[0]

    env = gym.make(args_cli.task, cfg=base_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load policy ONCE
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-13_21-02-41/model_999.pt'
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner.load(resume_path)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    export_policy_as_onnx(ppo_runner.alg.policy, path=export_model_dir, filename="policy.onnx")
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    eval_dt = (1 / 120)
    success_radius_m = 0.1
    device = env.unwrapped.device
    OFFSET_START = 4
    OFFSET_END = 7  # exclusive

    for goal_o in goal_orientations:
        for starting_o in starting_orientations:
            for vel in init_velocities:
                for offset in offsets:
                    # 1) Set cfg first (so reset sees non-None)
                    ok = _set_trial_cfg(env, goal_o=goal_o, starting_o=starting_o, vel=vel, offset=offset)
                    if not ok:
                        raise RuntimeError("Could not access task cfg to set trial parameters")

                    # 2) Make sure reset runs with inference mode OFF
                    with torch.inference_mode(False):
                        obs, _ = env.reset()

                    # 3) Per-trial state (created outside inference_mode)
                    if offset[0] == 0.5:
                        timeout_s = 2.0
                    elif offset[0] == 2.5:
                        timeout_s = 4.0
                    elif offset[0] == 5.5:
                        timeout_s = 7.0
                    else:
                        raise RuntimeError("Unexpected offset encountered")

                    times_to_stop = torch.full((num_envs,), -1.0, device=device)
                    elapsed = torch.zeros((num_envs,), device=device)
                    active = torch.ones((num_envs,), dtype=torch.bool, device=device)
                    total_energy = torch.zeros((num_envs,), device=device)

                    # 4) Rollout with no_grad (NOT inference_mode)
                    while simulation_app.is_running() and active.any():
                        with torch.no_grad():
                            actions = policy(obs)
                            total_energy += torch.sum(actions ** 2, dim=1) * eval_dt
                            obs, rews, _, _ = env.step(actions)

                            offsets_b = obs[:, OFFSET_START:OFFSET_END]
                            dists = torch.norm(offsets_b, dim=1)
                            lin_speeds = torch.norm(obs[:, 11:14], dim=1)

                            # update timers/masks
                            elapsed = elapsed + (eval_dt * active.float())
                            newly_success = (dists <= success_radius_m) & (lin_speeds <= 0.5) & active
                            times_to_stop[newly_success] = elapsed[newly_success]
                            active[newly_success] = False

                            newly_failed = (elapsed >= timeout_s) & active
                            active[newly_failed] = False

                    avg_energy = total_energy.mean().item()
                    successes = (times_to_stop >= 0).sum().item()
                    failures = num_envs - successes
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
                    print(f"Average energy usage: {avg_energy:.4f}")

                    append_results(
                        goal_o=goal_o, starting_o=starting_o, vel=vel, offset=offset,
                        num_envs=num_envs, task=args_cli.task, checkpoint=resume_path,
                        eval_dt=eval_dt, timeout_s=timeout_s, success_radius_m=success_radius_m,
                        successes=successes, failures=failures, avg_energy=avg_energy,
                        success_times=success_times,
                    )

                    gc.collect()

    env.close()
    del env, ppo_runner
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

if __name__ == "__main__":
    main()
