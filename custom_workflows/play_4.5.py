import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
#parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
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

from rsl_rl.runners import OnPolicyRunner

import numpy as np
import statistics
import torch
import math

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
#    env_cfg = parse_env_cfg(
#        args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric
#    )
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-10-23_16-30-25/model_700.pt' 

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(dir(ppo_runner))
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    save_path = os.path.join("source", "results", "rsl_rl", agent_cfg.experiment_name, agent_cfg.load_run, agent_cfg.load_checkpoint[:-3] + "_play")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    w = csv.writer(open(os.path.join(save_path, "output.csv"), 'w'), delimiter=',')
    print(f"[INFO]: Saving results into: {save_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    #export_policy_as_jit(
    #    ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    #)
    #export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")
    export_policy_as_onnx(ppo_runner.alg.policy, path=export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()

    record_obs = []

    print("Recording distance observations")

    # simulate environment
    while simulation_app.is_running() and len(record_obs) < 10000:

        if len(record_obs) % 100 == 0:
            print(f"At step {len(record_obs)}")
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rews, _, _ = env.step(actions)

            obs = obs[0]
            detached = obs.cpu().detach()
            detached = detached.numpy()
            record_obs.append(detached[3:6])
            rews = rews[0]

            distance = torch.norm(obs)

        w.writerow([rews.cpu().item(), distance.cpu().item()])

    # close the simulator
    env.close()
    data = [math.sqrt(arr[0]**2 + arr[1]**2 + arr[2]**2) for arr in record_obs]
    z_axis = [arr[2] for arr in record_obs]
    y_axis = [arr[1] for arr in record_obs]
    x_axis = [arr[0] for arr in record_obs]
    mean_value = statistics.mean(data)
    print(f"Mean distance: {mean_value}")

    median_value = statistics.median(data)
    print(f"Median distance: {median_value}")

    stdev_value = statistics.stdev(data)
    print(f"Standard deviation, overall distance: {stdev_value}")

    mean_value_z = statistics.mean(z_axis)
    print(f"Mean z distance: {mean_value_z}")

    median_value_z = statistics.median(z_axis)
    print(f"Median z distance: {median_value_z}")

    stdev_value_z = statistics.stdev(z_axis)
    print(f"Standard deviation, z-axis: {stdev_value_z}")

    mean_value_y = statistics.mean(y_axis)
    print(f"Mean y distance: {mean_value_y}")

    median_value_y = statistics.median(y_axis)
    print(f"Median y distance: {median_value_y}")

    stdev_value_y = statistics.stdev(y_axis)
    print(f"Standard deviation, y-axis: {stdev_value_y}")

    mean_value_x = statistics.mean(x_axis)
    print(f"Mean x distance: {mean_value_x}")

    median_value_x = statistics.median(x_axis)
    print(f"Median x distance: {median_value_x}")

    stdev_value_x = statistics.stdev(x_axis)
    print(f"Standard deviation, x-axis: {stdev_value_x}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
