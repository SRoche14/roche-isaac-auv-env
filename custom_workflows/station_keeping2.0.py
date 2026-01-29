import argparse

from isaaclab.app import AppLauncher

import cv2
import numpy as np

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate station keeping performance with a trained RL agent.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-WarpAUV-Direct-v1", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
# TODO: update to require passing the checkpoint path
#parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint")
parser.add_argument("--duration", type=float, default=120.0, help="Duration of station keeping test in seconds")

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
import time
import math
from typing import Dict, List, Tuple

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw] in radians."""
    w, x, y, z = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Calculate the angular distance between two quaternions in radians."""
    # Ensure quaternions are normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate dot product
    dot_product = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    
    # Angular distance in radians
    angle = 2 * np.arccos(dot_product)
    
    return angle

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert euler angles [roll, pitch, yaw] in radians to quaternion [w, x, y, z]."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    
    return np.array([w, x, y, z])

def analyze_station_keeping_performance(log_data: List[Dict]) -> Dict:
    """Analyze station keeping performance from logged data."""
    if not log_data:
        return {}
    
    # Extract data
    positions = np.array([entry['position'] for entry in log_data])
    orientations = np.array([entry['orientation'] for entry in log_data])
    velocities = np.array([entry['velocity'] for entry in log_data])
    distances = np.array([entry['distance_to_target'] for entry in log_data])
    orientation_errors = np.array([entry['orientation_error'] for entry in log_data])
    actions = np.array([entry['actions'] for entry in log_data])
    
    # Position analysis
    pos_std = np.std(positions, axis=0)
    pos_max_deviation = np.max(np.abs(positions), axis=0)
    pos_rms = np.sqrt(np.mean(positions**2, axis=0))
    
    # Orientation analysis
    orient_std = np.std(orientations, axis=0)
    orient_max_deviation = np.max(np.abs(orientations), axis=0)
    orient_rms = np.sqrt(np.mean(orientations**2, axis=0))
    
    # Velocity analysis
    vel_std = np.std(velocities, axis=0)
    vel_rms = np.sqrt(np.mean(velocities**2, axis=0))
    
    # Overall performance metrics
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)
    distance_std = np.std(distances)
    
    avg_orientation_error = np.mean(orientation_errors)
    max_orientation_error = np.max(orientation_errors)
    orientation_error_std = np.std(orientation_errors)
    
    # Action analysis
    action_std = np.std(actions, axis=0)
    action_rms = np.sqrt(np.mean(actions**2, axis=0))
    total_action_magnitude = np.sum(np.abs(actions), axis=0)
    
    # Convergence analysis (find when system stabilizes)
    # Consider system stable when distance < threshold for consecutive steps
    stability_threshold = 0.5  # meters
    stable_steps = 0
    convergence_step = -1
    
    for i, distance in enumerate(distances):
        if distance < stability_threshold:
            stable_steps += 1
            if stable_steps >= 50 and convergence_step == -1:  # 50 consecutive stable steps
                convergence_step = i
        else:
            stable_steps = 0
    
    return {
        'position_std': pos_std.tolist(),
        'position_max_deviation': pos_max_deviation.tolist(),
        'position_rms': pos_rms.tolist(),
        'orientation_std': orient_std.tolist(),
        'orientation_max_deviation': orient_max_deviation.tolist(),
        'orientation_rms': orient_rms.tolist(),
        'velocity_std': vel_std.tolist(),
        'velocity_rms': vel_rms.tolist(),
        'avg_distance_to_target': float(avg_distance),
        'max_distance_to_target': float(max_distance),
        'distance_std': float(distance_std),
        'avg_orientation_error': float(avg_orientation_error),
        'max_orientation_error': float(max_orientation_error),
        'orientation_error_std': float(orientation_error_std),
        'action_std': action_std.tolist(),
        'action_rms': action_rms.tolist(),
        'total_action_magnitude': total_action_magnitude.tolist(),
        'convergence_step': convergence_step,
        'convergence_time': convergence_step * 0.00833 if convergence_step != -1 else -1,  # Assuming 120Hz sim
        'stability_percentage': float(np.sum(distances < stability_threshold) / len(distances) * 100)
    }

def main():
    # Parse environment and agent config
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Modify environment for station keeping
    env_cfg.episode_length_s = args_cli.duration
    env_cfg.eval_mode = True  # Disable randomization for consistent testing
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    # Set random seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # Load trained policy
    # TODO: change to use dynamic checkpoint
    resume_path = '/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-08-14_14-53-43/model_3750.pt'
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Export policy (optional)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(ppo_runner.alg.policy, path=export_model_dir, filename="policy.onnx")

    # --- STATION-KEEPING PARAMETERS ---
    # We'll get the actual goal from the environment (the red dot)
    
    # Create output directory
    output_dir = f"station_keeping_results_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "station_keeping_log.csv")
    analysis_file = os.path.join(output_dir, "performance_analysis.txt")
    
    print(f"[INFO]: Will evaluate station keeping at the environment's goal (red dot)")
    print(f"[INFO]: Test duration: {args_cli.duration} seconds")
    print(f"[INFO]: Results will be saved to: {output_dir}")

    # CSV header
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "env_id", "timestamp", "x", "y", "z", "roll", "pitch", "yaw",
            "distance_to_goal", "orientation_error_to_goal", "vx", "vy", "vz",
            "wx", "wy", "wz", "action_0", "action_1", "action_2", "action_3", "action_4", "action_5"
        ])

    # Reset environment
    obs, _ = env.get_observations()
    
    # Note: The environment expects orientation goals, not position goals
    # For proper station keeping, the environment should be modified to support position goals
    
    log_data = []
    step_count = 0
    start_time = time.time()
    
    print("[INFO]: Starting station keeping evaluation...")
    
    while simulation_app.is_running() and (time.time() - start_time) < args_cli.duration:
        # Step the simulator
        k = cv2.waitKey(1)
        if k == 27:  # ESC key
            break

        # Run in inference mode
        with torch.inference_mode():
            # Get action from policy
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)

            # --- EXTRACT AND LOG STATION-KEEPING METRICS ---
            # Parse observations for ALL environments
            num_envs = obs.shape[0]
            
            # Process each environment
            for env_idx in range(num_envs):
                # Parse observation for this environment
                goal_quat = obs[env_idx, :4].cpu().numpy()  # [0:4] - goal orientation (quaternion)
                offset_body = obs[env_idx, 4:7].cpu().numpy()  # [4:7] - position offset in body frame
                current_quat = obs[env_idx, 7:11].cpu().numpy()  # [7:11] - current orientation (quaternion)
                current_lin_vel = obs[env_idx, 11:14].cpu().numpy()  # [11:14] - linear velocity in body frame
                current_ang_vel = obs[env_idx, 14:17].cpu().numpy()  # [14:17] - angular velocity in body frame
                
                # Calculate actual position from the offset (relative to environment origin)
                # The offset_body is already the position relative to the environment origin
                current_pos = offset_body
                
                # Convert quaternions to euler angles for easier analysis
                current_euler = quaternion_to_euler(current_quat)
                
                # Calculate metrics using the ACTUAL GOAL from the environment (red dot)
                # The goal_quat represents where the robot is trying to go
                dist_to_target = np.linalg.norm(current_pos)  # Distance from origin (where goal is)
                
                # Calculate orientation error using proper quaternion distance to the goal
                orientation_error = quaternion_distance(current_quat, goal_quat)
                
                # Get actions for this environment
                action_values = actions[env_idx].cpu().numpy()
                
                # Log data for this environment
                log_entry = {
                    'step': step_count,
                    'env_id': env_idx,
                    'timestamp': time.time() - start_time,
                    'position': current_pos,
                    'orientation': current_euler,
                    'velocity': current_lin_vel,
                    'angular_velocity': current_ang_vel,
                    'distance_to_target': dist_to_target,
                    'orientation_error': orientation_error,
                    'actions': action_values
                }
                log_data.append(log_entry)
                
                # Write to CSV for this environment
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        step_count, env_idx, time.time() - start_time,
                        *current_pos, *current_euler,
                        dist_to_target, orientation_error,
                        *current_lin_vel, *current_ang_vel,
                        *action_values
                    ])

            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                # Calculate aggregate statistics across all environments for this step
                step_distances = []
                step_orientation_errors = []
                
                for env_idx in range(num_envs):
                    env_offset = obs[env_idx, 4:7].cpu().numpy()
                    env_quat = obs[env_idx, 7:11].cpu().numpy()
                    env_goal_quat = obs[env_idx, :4].cpu().numpy()
                    
                    env_dist = np.linalg.norm(env_offset)
                    env_orient_error = quaternion_distance(env_quat, env_goal_quat)
                    
                    step_distances.append(env_dist)
                    step_orientation_errors.append(env_orient_error)
                
                avg_distance = np.mean(step_distances)
                avg_orientation_error = np.mean(step_orientation_errors)
                max_distance = np.max(step_distances)
                max_orientation_error = np.max(step_orientation_errors)
                
                print(f"[INFO]: Step {step_count}, {num_envs} environments")
                print(f"       Avg Distance: {avg_distance:.3f}m, Max: {max_distance:.3f}m")
                print(f"       Avg Orientation Error: {avg_orientation_error:.3f}rad ({avg_orientation_error*180/np.pi:.1f}°), Max: {max_orientation_error*180/np.pi:.1f}°")

            # Optional: reset environment if done
            if dones[0]:
                obs, _ = env.reset()
                print("[INFO]: Environment reset")

    # Analyze performance
    print("[INFO]: Analyzing station keeping performance...")
    performance_metrics = analyze_station_keeping_performance(log_data)
    
    # Save analysis results
    with open(analysis_file, "w") as f:
        f.write("STATION KEEPING PERFORMANCE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Configuration:\n")
        f.write(f"  Evaluating station keeping at environment goal (red dot)\n")
        f.write(f"  Number of Environments: {num_envs}\n")
        f.write(f"  Duration: {args_cli.duration} seconds\n")
        f.write(f"  Total Steps: {step_count}\n")
        f.write(f"  Total Data Points: {step_count * num_envs}\n\n")
        
        f.write(f"Position Performance:\n")
        f.write(f"  Average Distance to Goal: {performance_metrics['avg_distance_to_target']:.4f} m\n")
        f.write(f"  Max Distance to Goal: {performance_metrics['max_distance_to_target']:.4f} m\n")
        f.write(f"  Distance Standard Deviation: {performance_metrics['distance_std']:.4f} m\n")
        f.write(f"  Position RMS: {performance_metrics['position_rms']}\n")
        f.write(f"  Position Standard Deviation: {performance_metrics['position_std']}\n\n")
        
        f.write(f"Orientation Performance:\n")
        f.write(f"  Average Orientation Error to Goal: {performance_metrics['avg_orientation_error']:.4f} rad\n")
        f.write(f"  Max Orientation Error to Goal: {performance_metrics['max_orientation_error']:.4f} rad\n")
        f.write(f"  Orientation Error Standard Deviation: {performance_metrics['orientation_error_std']:.4f} rad\n")
        f.write(f"  Orientation RMS: {performance_metrics['orientation_rms']}\n\n")
        
        f.write(f"Velocity Performance:\n")
        f.write(f"  Linear Velocity RMS: {performance_metrics['velocity_rms']}\n")
        f.write(f"  Linear Velocity Standard Deviation: {performance_metrics['velocity_std']}\n\n")
        
        f.write(f"Control Performance:\n")
        f.write(f"  Action RMS: {performance_metrics['action_rms']}\n")
        f.write(f"  Action Standard Deviation: {performance_metrics['action_std']}\n")
        f.write(f"  Total Action Magnitude: {performance_metrics['total_action_magnitude']}\n\n")
        
        f.write(f"Convergence Analysis:\n")
        if performance_metrics['convergence_step'] != -1:
            f.write(f"  Convergence Time: {performance_metrics['convergence_time']:.2f} seconds\n")
            f.write(f"  Convergence Step: {performance_metrics['convergence_step']}\n")
        else:
            f.write(f"  System did not converge within stability threshold\n")
        f.write(f"  Stability Percentage: {performance_metrics['stability_percentage']:.1f}%\n")

    # Print summary
    print(f"\n[INFO]: Station keeping evaluation completed!")
    print(f"[INFO]: Results saved to: {output_dir}")
    print(f"[INFO]: Evaluated {num_envs} environments over {step_count} steps")
    print(f"[INFO]: Final aggregate statistics:")
    
    # Calculate final aggregate statistics
    final_distances = []
    final_orientation_errors = []
    
    for env_idx in range(num_envs):
        env_offset = obs[env_idx, 4:7].cpu().numpy()
        env_quat = obs[env_idx, 7:11].cpu().numpy()
        env_goal_quat = obs[env_idx, :4].cpu().numpy()
        
        env_dist = np.linalg.norm(env_offset)
        env_orient_error = quaternion_distance(env_quat, env_goal_quat)
        
        final_distances.append(env_dist)
        final_orientation_errors.append(env_orient_error)
    
    avg_final_distance = np.mean(final_distances)
    avg_final_orientation_error = np.mean(final_orientation_errors)
    max_final_distance = np.max(final_distances)
    max_final_orientation_error = np.max(final_orientation_errors)
    
    print(f"       Avg Distance to Goal: {avg_final_distance:.4f} m, Max: {max_final_distance:.4f} m")
    print(f"       Avg Orientation Error: {avg_final_orientation_error:.4f} rad ({avg_final_orientation_error*180/np.pi:.1f}°), Max: {max_final_orientation_error*180/np.pi:.1f}°")
    print(f"[INFO]: Stability: {performance_metrics['stability_percentage']:.1f}%")
    
    if performance_metrics['convergence_step'] != -1:
        print(f"[INFO]: System converged in {performance_metrics['convergence_time']:.2f} seconds")
    else:
        print(f"[INFO]: System did not converge within stability threshold")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()


