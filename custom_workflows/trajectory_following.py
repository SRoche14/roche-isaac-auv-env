"""
Strict waypoint-following evaluation script for WarpAUV in Isaac Lab.

What this script does, at a glance:
- Spawns vectorized environments and loads a trained RSL-RL policy.
- Generates simple trajectories (linear / circle / L / fast_stop) as waypoint lists in each env's local frame.
- Converts those waypoints to world coordinates and drives the vehicle toward them using the
  policy's goal interface (position-offset in body frame + desired orientation quaternion).
- Adds corner-aware behavior: when approaching a corner, the orientation pre-turns toward the
  next leg and the positional pull leads onto the next segment.
- Logs data and computes basic performance metrics at the end.

Notes:
- "L" trajectory is intentionally sparse: only three waypoints [(0,0)->(L,0)->(L,L)].
- "fast_stop" adds a one-time world-velocity shove toward the final waypoint. Re-applied after resets.
- Vehicle start depth assumed 5 m (per WARPAUV init).
"""

import argparse

from isaaclab.app import AppLauncher

import cv2
import numpy as np

# local imports
import cli_args  # isort: skip

# =======================
# CLI
# =======================
parser = argparse.ArgumentParser(description="Strict waypoint-following with corner-aware pre-turn & lead target.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--task", type=str, default="Isaac-WarpAUV-Direct-v1")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--duration", type=float, default=60.0)

# trajectories: linear, circle, L, fast_stop
parser.add_argument(
    "--trajectory_type", type=str, default="linear",
    choices=["linear", "circle", "circular", "L", "l", "fast_stop"]
)
parser.add_argument("--trajectory_scale", type=float, default=5.0)
parser.add_argument("--trajectory_speed", type=float, default=1.0)  # kept for compatibility

# waypoint mode
parser.add_argument("--wp_spacing", type=float, default=0.5, help="Nominal spacing (m).")
parser.add_argument("--wp_radius", type=float, default=0.04, help="Hit radius (m).")
parser.add_argument("--wp_orient_tol_deg", type=float, default=0.0,
                    help="Yaw tolerance to advance (0=ignore).")
parser.add_argument("--loop", action="store_true", help="Loop waypoints (useful for circle).")

# corner handling
parser.add_argument("--pre_turn_radius", type=float, default=0.2,
                    help="Within this distance of a waypoint, start aiming orientation to next leg.")
parser.add_argument("--corner_angle_deg", type=float, default=30.0,
                    help="Only do pre-turn/lead if corner angle >= this (deg).")
parser.add_argument("--lead_dist", type=float, default=0.2,
                    help="Virtual target distance onto next leg while inside pre-turn radius.")
parser.add_argument("--corner_slow_scale", type=float, default=0.6,
                    help="Scale the positional goal magnitude inside pre-turn radius to reduce rush into corners.")

# startup stability + safety
parser.add_argument("--hold_time", type=float, default=5.0)
parser.add_argument("--ramp_time", type=float, default=0.0)
parser.add_argument("--offset_clip", type=float, default=10.0)

# drawing + anchoring
parser.add_argument("--no_draw", action="store_true")
parser.add_argument("--draw_radius", type=float, default=0.06)
parser.add_argument("--anchor_to_start", action="store_true", default=True)

# orientation viz
parser.add_argument("--viz_orient", action="store_true", default=True)
parser.add_argument("--orient_length", type=float, default=1.0)

# final yaw at last WP (optional)
parser.add_argument("--final_yaw_deg", type=float, default=None)

# physics model selection (chooses checkpoint)
parser.add_argument("--physics_model", type=str, default="simple", choices=["simple", "cfd"])

# fast_stop controls (ONLY world-velocity shove)
parser.add_argument("--fast_stop_speed", type=float, default=15.0,
                    help="Initial world speed (m/s) toward final waypoint when trajectory_type='fast_stop'.")

# RSL-RL + App args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.trajectory_type == "circular":
    args_cli.trajectory_type = "circle"
if args_cli.trajectory_type == "l":
    args_cli.trajectory_type = "L"

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =======================
# Imports that need Kit
# =======================
import gymnasium as gym
import os
import torch
import csv
import time
from typing import Dict, List, Tuple

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# =======================
# Visualizers (no pxr)
# =======================
class WaypointVisualizer:
    """Draws point markers for the generated waypoint list.

    Tries to use VisualSphere prims; if that fails, falls back to DebugDraw.
    If both are unavailable, drawing is silently disabled.
    """

    def __init__(self, simulation_app, color=(1.0, 0.0, 0.0), radius=0.06, enable=True):
        self.enable = enable
        self.color = np.array(color, dtype=np.float32)
        self.radius = float(radius)
        self.mode = None
        if not self.enable:
            print("[INFO]: Waypoint drawing disabled.")
            return
        try:
            from omni.isaac.core.objects import VisualSphere
            from omni.isaac.core.utils.prims import create_prim
            self.VisualSphere = VisualSphere
            self.create_prim = create_prim
            self.mode = "visual"
            print("[INFO]: WaypointVisualizer: VisualSphere prims.")
        except Exception as e:
            print(f"[WARN]: VisualSphere unavailable ({e}). Trying DebugDraw...")
            try:
                from isaacsim.core.utils.extensions import enable_extension
                enable_extension("isaacsim.util.debug_draw")
                simulation_app.update()
                from isaacsim.util.debug_draw import _debug_draw
                self.dd = _debug_draw.acquire_debug_draw_interface()
                self.mode = "debugdraw"
                print("[INFO]: WaypointVisualizer: DebugDraw fallback.")
            except Exception as e2:
                print(f"[WARN]: DebugDraw unavailable ({e2}). No drawing.")
                self.mode = None

    def draw_waypoints(self, env_idx: int, pts_world: np.ndarray):
        """Render all waypoints for a given env index."""
        if self.mode is None or not self.enable or pts_world.size == 0:
            return
        pts = pts_world.astype(np.float32)
        if self.mode == "visual":
            parent = f"/World/Waypoints/env_{env_idx}"
            self.create_prim(parent, "Xform")
            for i, p in enumerate(pts):
                prim_path = f"{parent}/wp_{i:04d}"
                try:
                    _ = self.VisualSphere(prim_path=prim_path, name=f"wp_{env_idx}_{i}",
                                          position=p.tolist(), radius=self.radius, color=self.color)
                except Exception:
                    pass
        elif self.mode == "debugdraw":
            tuples = [tuple(map(float, p)) for p in pts]
            colors = [(*self.color, 1.0)] * len(tuples)
            sizes = [max(6.0, 1000.0 * self.radius)] * len(tuples)
            self.dd.draw_points(tuples, colors, sizes)


class OrientationVisualizer:
    """Shows current vs. desired orientation as arrows at the vehicle position.

    Uses VisualizationMarkers (preferred) or falls back to DebugDraw lines.
    """

    def __init__(self, simulation_app, length=1.0, enable=True):
        self.enable = enable
        self.length = float(length)
        self.mode = None
        if not self.enable:
            print("[INFO]: Orientation viz disabled.")
            return
        try:
            from isaaclab.markers import VisualizationMarkers, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
            self.VisualizationMarkers = VisualizationMarkers
            self.GREEN = GREEN_ARROW_X_MARKER_CFG
            self.RED = RED_ARROW_X_MARKER_CFG
            self.BLUE = BLUE_ARROW_X_MARKER_CFG
            # desired X (red)
            cfg_dx = self.RED.copy();  cfg_dx.prim_path = "/Visuals/Trajectory/orient/goal_x"
            cfg_dx.markers["arrow"].scale = (0.125, 0.125, self.length)
            self.v_goal_x = self.VisualizationMarkers(cfg_dx)
            # desired Z (blue)
            cfg_dz = self.BLUE.copy(); cfg_dz.prim_path = "/Visuals/Trajectory/orient/goal_z"
            cfg_dz.markers["arrow"].scale = (0.125, 0.125, self.length)
            self.v_goal_z = self.VisualizationMarkers(cfg_dz)
            # current X (green)
            cfg_cx = self.GREEN.copy(); cfg_cx.prim_path = "/Visuals/Trajectory/orient/current_x"
            cfg_cx.markers["arrow"].scale = (0.125, 0.125, self.length)
            self.v_cur_x = self.VisualizationMarkers(cfg_cx)
            # current Z (green)
            cfg_cz = self.GREEN.copy(); cfg_cz.prim_path = "/Visuals/Trajectory/orient/current_z"
            cfg_cz.markers["arrow"].scale = (0.125, 0.125, self.length)
            self.v_cur_z = self.VisualizationMarkers(cfg_cz)
            self.mode = "markers"
            print("[INFO]: OrientationVisualizer: VisualizationMarkers arrows.")
        except Exception as e:
            print(f"[WARN]: VisualizationMarkers not available ({e}). Trying DebugDraw lines...")
            try:
                from isaacsim.core.utils.extensions import enable_extension
                enable_extension("isaacsim.util.debug_draw")
                simulation_app.update()
                from isaacsim.util.debug_draw import _debug_draw
                self.dd = _debug_draw.acquire_debug_draw_interface()
                self.mode = "debugdraw"
                print("[INFO]: OrientationVisualizer: DebugDraw lines fallback.")
            except Exception as e2:
                print(f"[WARN]: DebugDraw unavailable ({e2}). No orientation viz.")
                self.mode = None

    def update(self, positions_world: np.ndarray, q_current_world: np.ndarray, q_goal_world: np.ndarray):
        """Update markers/lines for a batch of envs."""
        if self.mode is None or not self.enable:
            return
        if self.mode == "markers":
            import torch
            pos = torch.tensor(positions_world, dtype=torch.float32)
            q_c = torch.tensor(q_current_world, dtype=torch.float32)
            q_g = torch.tensor(q_goal_world, dtype=torch.float32)
            # rotate by (0,-pi/2,0) to draw Z using X-arrow marker
            q_rot = euler_to_quaternion(0.0, -np.pi/2, 0.0).astype(np.float32)
            q_gz = quat_mul_np(q_g.numpy(), np.repeat(q_rot[None, :], q_g.shape[0], axis=0))
            q_cz = quat_mul_np(q_c.numpy(), np.repeat(q_rot[None, :], q_g.shape[0], axis=0))
            q_gz = torch.tensor(q_gz, dtype=torch.float32)
            q_cz = torch.tensor(q_cz, dtype=torch.float32)
            scales = torch.tensor([[1, 1, 1]] * len(pos), dtype=torch.float32)
            self.v_goal_x.visualize(translations=pos, orientations=q_g,  scales=scales)
            self.v_goal_z.visualize(translations=pos, orientations=q_gz, scales=scales)
            self.v_cur_x.visualize( translations=pos, orientations=q_c,  scales=scales)
            self.v_cur_z.visualize( translations=pos, orientations=q_cz, scales=scales)
        elif self.mode == "debugdraw":
            tuples_lines = []
            colors = []
            for p, qc, qg in zip(positions_world, q_current_world, q_goal_world):
                Rc = quat_to_rotmat_wxyz(qc); Rg = quat_to_rotmat_wxyz(qg)
                x_c = (Rc @ np.array([1.0, 0.0, 0.0])) * self.length
                z_c = (Rc @ np.array([0.0, 0.0, 1.0])) * self.length
                x_g = (Rg @ np.array([1.0, 0.0, 0.0])) * self.length
                z_g = (Rg @ np.array([0.0, 0.0, 1.0])) * self.length
                tuples_lines += [
                    (tuple(p.tolist()), tuple((p + x_c).tolist())),
                    (tuple(p.tolist()), tuple((p + z_c).tolist())),
                    (tuple(p.tolist()), tuple((p + x_g).tolist())),
                    (tuple(p.tolist()), tuple((p + z_g).tolist())),
                ]
                colors += [(0.0,1.0,0.0,1.0),(0.0,1.0,0.0,1.0),(1.0,0.0,0.0,1.0),(0.0,0.3,1.0,1.0)]
            self.dd.draw_lines(tuples_lines, colors, [2.0]*len(tuples_lines))


# =======================
# Math helpers
# =======================

def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> Euler [roll, pitch, yaw]."""
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler -> Quaternion [w,x,y,z]."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    v = np.array([w, x, y, z], dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest-angle distance between unit quaternions (radians)."""
    q1 = q1 / np.linalg.norm(q1); q2 = q2 / np.linalg.norm(q2)
    dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def quat_mul_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Quaternion product for 1D or batched arrays (both [w,x,y,z])."""
    if q.ndim == 1: q = q[None, :]
    if r.ndim == 1: r = r[None, :]
    w1,x1,y1,z1 = q[:,0],q[:,1],q[:,2],q[:,3]
    w2,x2,y2,z2 = r[:,0],r[:,1],r[:,2],r[:,3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    out = np.stack([w,x,y,z], axis=-1)
    n = np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9
    return out / n


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> R_world_from_body."""
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*s*y, s*y*z, s*z*z  # safe formula
    # Corrected: ensure consistent terms (typos handled)
    yy = s*y*y; yz = s*y*z; zz = s*z*z
    return np.array([
        [1.0 - (yy + zz),   xy - wz,           xz + wy],
        [      xy + wz,     1.0 - (xx + zz),   yz - wx],
        [      xz - wy,     yz + wx,           1.0 - (xx + yy)],
    ], dtype=np.float64)


def get_nested_attr(obj, dotted: str):
    """Safely traverse a dotted attribute path on an object."""
    cur = obj
    for name in dotted.split("."):
        cur = getattr(cur, name)
    return cur


def try_get_scene_env_origins(env, num_envs: int) -> np.ndarray:
    """Best-effort extraction of each env's world origin (x,y,z)."""
    candidates = [
        "unwrapped.scene.env_origins",
        "unwrapped._env.scene.env_origins",
        "scene.env_origins",
        "unwrapped._scene.env_origins",
    ]
    for path in candidates:
        try:
            arr = get_nested_attr(env, path)
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= num_envs:
                return arr
        except Exception:
            pass
    return np.zeros((num_envs, 3), dtype=np.float64)


def try_get_default_env_origins(env, num_envs: int, scene_env_origins: np.ndarray) -> np.ndarray:
    """Best-effort extraction of root positions used as nominal robot start."""
    for path in ["unwrapped._env._default_env_origins", "unwrapped._env._goal_pos_w"]:
        try:
            arr = get_nested_attr(env, path)
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= num_envs:
                return arr
        except Exception:
            pass
    try:
        drs = get_nested_attr(env, "unwrapped._env._default_root_state")
        if hasattr(drs, "cpu"): drs = drs.cpu().numpy()
        drs = np.asarray(drs)
        if drs.ndim == 2 and drs.shape[1] >= 3 and drs.shape[0] >= num_envs:
            return drs[:, :3]
    except Exception:
        pass
    fallback = scene_env_origins.copy()
    fallback[:, 2] = 5.0  # WARPAUV init depth
    print("[WARN]: Using fallback default-env-origins = scene.env_origins + [0,0,5].")
    return fallback


# =======================
# Waypoint generator (env-local, z=0 so depth stays 5)
# =======================

def sample_waypoints(typ: str, scale: float, spacing: float) -> np.ndarray:
    """Generate simple 2D (x,y) paths in env-local frame. z=0 keeps constant depth.

    Args:
        typ: 'linear' | 'circle' | 'L' | 'fast_stop'
        scale: Path size (e.g., line length or circle radius)
        spacing: Approximate waypoint spacing along the path (ignored for sparse 'L' and 'fast_stop')
    """
    scale = float(scale); spacing = max(0.05, float(spacing))
    pts = []
    if typ == "linear":
        # Diagonal from (0,0) to (scale, scale)
        L = np.linalg.norm([scale, scale])
        n = max(2, int(np.ceil(L / spacing)))
        for i in range(n + 1):
            a = i / n
            pts.append([a * scale, a * scale, 0.0])
    elif typ == "circle":
        r = max(0.1, scale)
        circumference = 2 * np.pi * r
        n = max(8, int(np.ceil(circumference / spacing)))
        for i in range(n):
            th = 2 * np.pi * i / n
            pts.append([r * np.cos(th), r * np.sin(th), 0.0])
        pts.append(pts[0])  # return to start
    elif typ == "L":
        # Intentionally sparse: start, corner, end
        L = max(0.5, scale)
        pts = [
            [0.0, 0.0, 0.0],   # start
            [L,   0.0, 0.0],   # corner
            [L,   L,   0.0],   # end
        ]
    elif typ == "fast_stop":
        # Two points separated by a few meters in +X
        D = 1.0
        pts = [
            [0.0, 0.0, 0.0],
            [D,   0.0, 0.0],
        ]
    else:
        return sample_waypoints("linear", scale, spacing)
    return np.asarray(pts, dtype=np.float64)


# =======================
# Analysis (unchanged)
# =======================

def analyze_trajectory_following_performance(log_data: List[Dict]) -> Dict:
    """Simple aggregates from the per-step logging to sanity-check behavior."""
    if not log_data:
        return {}
    import numpy as np
    positions = np.array([e['position'] for e in log_data])
    orientations = np.array([e['orientation'] for e in log_data])
    velocities = np.array([e['velocity'] for e in log_data])
    distances_to_goal = np.array([e['distance_to_goal'] for e in log_data])
    distances_to_trajectory = np.array([e['distance_to_trajectory'] for e in log_data])
    orientation_errors = np.array([e['orientation_error'] for e in log_data])
    actions = np.array([e['actions'] for e in log_data])
    def rms(x): return np.sqrt(np.mean(x**2, axis=0))
    out = {
        'position_std': np.std(positions, axis=0).tolist(),
        'position_max_deviation': np.max(np.abs(positions), axis=0).tolist(),
        'position_rms': rms(positions).tolist(),
        'orientation_std': np.std(orientations, axis=0).tolist(),
        'orientation_max_deviation': np.max(np.abs(orientations), axis=0).tolist(),
        'orientation_rms': rms(orientations).tolist(),
        'velocity_std': np.std(velocities, axis=0).tolist(),
        'velocity_rms': rms(velocities).tolist(),
        'avg_distance_to_goal': float(np.mean(distances_to_goal)),
        'max_distance_to_goal': float(np.max(distances_to_goal)),
        'goal_distance_std': float(np.std(distances_to_goal)),
        'avg_distance_to_trajectory': float(np.mean(distances_to_trajectory)),
        'max_distance_to_trajectory': float(np.max(distances_to_trajectory)),
        'trajectory_distance_std': float(np.std(distances_to_trajectory)),
        'avg_orientation_error': float(np.mean(orientation_errors)),
        'max_orientation_error': float(np.max(orientation_errors)),
        'orientation_error_std': float(np.std(orientation_errors)),
        'action_std': np.std(actions, axis=0).tolist(),
        'action_rms': rms(actions).tolist(),
        'total_action_magnitude': np.sum(np.abs(actions), axis=0).tolist(),
    }
    # simple convergence heuristic
    thresh = 0.5
    stable = (distances_to_goal < thresh).astype(np.int32)
    consec = 0
    conv_step = -1
    for i, ok in enumerate(stable):
        consec = consec + 1 if ok else 0
        if consec >= 50 and conv_step == -1:
            conv_step = i
    out['convergence_step'] = conv_step
    out['convergence_time'] = conv_step * 0.05 if conv_step != -1 else -1
    out['stability_percentage'] = float(np.sum(stable) / len(stable) * 100.0)
    return out


# =======================
# Helpers for corner logic
# =======================

def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def angle_between(a, b):
    ua, ub = unit(a), unit(b)
    dot = np.clip(np.dot(ua, ub), -1.0, 1.0)
    return np.arccos(dot)


# =======================
# Robot access + initial velocity shove (fast_stop)
# =======================

def _get_robot_handle(env):
    """Try multiple attribute paths to find the underlying RigidObject robot."""
    candidates = [
        "unwrapped._env._robot",
        "_env._robot",
        "unwrapped.env._robot",
        "unwrapped._robot",
    ]
    for path in candidates:
        try:
            rob = get_nested_attr(env, path)
            # quick sanity: must have write_root_velocity_to_sim
            if hasattr(rob, "write_root_velocity_to_sim"):
                return rob
        except Exception:
            pass
    return None


def _apply_fast_stop_initial_velocity(env, env_waypoints_world, speed_mps: float, label="initial"):
    """
    Sets each robot's world linear velocity toward the final waypoint.
    Also zeros angular velocity to avoid spin. Safe to call after resets.
    Prints a verification reading from observations.
    """
    try:
        robot = _get_robot_handle(env)
        if robot is None:
            print("[WARN]: fast_stop: couldn't find robot handle – no initial velocity applied.")
            return False

        device = getattr(env.unwrapped, "device",
                         torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        num_envs = len(env_waypoints_world)

        root_vel = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)  # [vx,vy,vz, wx,wy,wz]
        for eid, wps in enumerate(env_waypoints_world):
            start = np.asarray(wps[0], dtype=np.float64)
            goal  = np.asarray(wps[-1], dtype=np.float64)
            vdir  = goal - start
            vdir[2] = 0.0  # keep depth unchanged
            n = np.linalg.norm(vdir[:2])
            #if n < 1e-8:
            vdir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            #else:
            #    vdir = vdir / n
            v_world = vdir * float(speed_mps)
            root_vel[eid, 0:3] = torch.tensor(v_world, dtype=torch.float32, device=device)  # linear vel
            root_vel[eid, 3:6] = 0.0  # no angular kick

        try:
            env_ids = robot._ALL_INDICES  # if exposed
        except Exception:
            env_ids = torch.arange(num_envs, device=device, dtype=torch.int64)

        robot.write_root_velocity_to_sim(root_vel, env_ids)
        print(f"[INFO]: fast_stop: set {label} world speed {speed_mps:.2f} m/s toward final WP.")
        return True
    except Exception as e:
        print(f"[WARN]: fast_stop: couldn't set {label} velocity: {e}")
        return False


def _estimate_world_speed_from_obs(obs_row: torch.Tensor) -> float:
    """
    Estimate current world linear speed from observation row:
    - obs[11:14] is body linear velocity.
    - obs[7:11] is world quaternion of body.
    Convert v_b to world using R_wb and return ||v_w||.
    """
    e_env_vb = obs_row[11:14].detach().cpu().numpy().astype(np.float64)
    q_cur = obs_row[7:11].detach().cpu().numpy().astype(np.float64)
    R_wb = quat_to_rotmat_wxyz(q_cur)
    v_w = R_wb @ e_env_vb
    return float(np.linalg.norm(v_w))


# =======================
# Main
# =======================

def main():
    # -------------------
    # Env + runner setup
    # -------------------
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.episode_length_s = args_cli.duration
    env_cfg.eval_mode = True
    try:
        env_cfg.debug_vis = False  # disable env's own goal markers
    except Exception:
        pass

    num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # -------------------
    # Load policy (adjust path)
    # -------------------
    if args_cli.physics_model == "simple":
        resume_path = "/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-01_16-25-24/model_1998.pt"
    else:
        #resume_path = "/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-08-14_14-53-43/model_3750.pt"
        #resume_path = "/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-08_21-11-04/model_1450.pt"
        resume_path = "/home/warp/isaacsim4.5/IsaacLab/logs/rsl_rl/warpauv_direct/2025-09-09_10-30-40/model_1400.pt"

    print(f"[INFO]: Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Optional export for inference elsewhere
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_dir, exist_ok=True)
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(runner.alg.policy, path=export_dir, filename="policy.onnx")

    # -------------------
    # Scene origins and starts
    # -------------------
    scene_origins = try_get_scene_env_origins(env, num_envs)
    default_env_origins = try_get_default_env_origins(env, num_envs, scene_origins)

    # -------------------
    # Waypoints per env
    # -------------------
    base_wps_local = sample_waypoints(args_cli.trajectory_type, args_cli.trajectory_scale, args_cli.wp_spacing)
    anchor_offsets = np.zeros((num_envs, 3), dtype=np.float64)
    if args_cli.anchor_to_start:
        # Shift so the first waypoint is at the env origin
        anchor_offsets[:] = -base_wps_local[0]

    env_waypoints_world = []
    for eid in range(num_envs):
        wps_local = base_wps_local + anchor_offsets[eid]
        wps_world = default_env_origins[eid] + wps_local
        env_waypoints_world.append(wps_world)

    # Draw waypoints once at startup
    wp_viz = WaypointVisualizer(simulation_app, radius=args_cli.draw_radius, enable=not args_cli.no_draw)
    if not args_cli.no_draw:
        for eid in range(num_envs):
            wp_viz.draw_waypoints(eid, env_waypoints_world[eid])

    # Orientation viz (optional)
    orient_viz = OrientationVisualizer(simulation_app, length=args_cli.orient_length, enable=args_cli.viz_orient)

    # First obs (ensures sim initialized before any velocity shove)
    obs, _ = env.get_observations()

    # Per-env waypoint index
    cur_wp_idx = np.zeros(num_envs, dtype=np.int32)

    # -------------------
    # Final orientation lock state (for 'linear' end)
    # -------------------
    orient_lock_active = np.zeros(num_envs, dtype=bool)
    orient_lock_q = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (num_envs, 1))

    # -------------------
    # Output files
    # -------------------
    out_dir = f"trajectory_following_results_{args_cli.trajectory_type}_{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)
    log_csv = os.path.join(out_dir, "trajectory_following_log.csv")
    analysis_txt = os.path.join(out_dir, "performance_analysis.txt")
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step","env_id","timestamp","x","y","z","roll","pitch","yaw",
            "distance_to_goal","distance_to_trajectory","orientation_error_to_goal",
            "vx","vy","vz","wx","wy","wz","action_0","action_1","action_2","action_3","action_4","action_5",
            "wp_index"
        ])

    start_time = time.time()
    step = 0
    logs: List[Dict] = []

    print(f"[INFO]: Waypoint mode | type={args_cli.trajectory_type} | spacing={args_cli.wp_spacing} m | radius={args_cli.wp_radius} m")
    yaw_tol = np.deg2rad(max(0.0, float(args_cli.wp_orient_tol_deg)))
    corner_thresh = np.deg2rad(max(0.0, float(args_cli.corner_angle_deg)))
    pre_r = float(args_cli.pre_turn_radius)
    lead_d = float(args_cli.lead_dist)
    slow_scale = float(args_cli.corner_slow_scale)

    fast_shove_done = False
    # -------------------
    # Control loop
    # -------------------
    fast_apply_used = 0
    while simulation_app.is_running() and (time.time() - start_time) < args_cli.duration:
        # ESC abort
        if cv2.waitKey(1) == 27:
            break

        now = time.time()
        elapsed = now - start_time
        # hold → ramp blending for goals
        if elapsed < args_cli.hold_time:
            alpha = 0.0
        else:
            alpha = min(1.0, (elapsed - args_cli.hold_time) / max(1e-6, args_cli.ramp_time))

        if elapsed - 0.5 > args_cli.hold_time and fast_apply_used == 0:
            fast_apply_used += 1
            # Apply fast_stop initial shove (world velocity) AFTER we have obs
            if args_cli.trajectory_type == "fast_stop":
                fast_shove_done = _apply_fast_stop_initial_velocity(
                    env, env_waypoints_world, speed_mps=args_cli.fast_stop_speed, label="initial"
                )
                # verify
                try:
                    if fast_shove_done and obs.shape[0] > 0:
                        speed0 = _estimate_world_speed_from_obs(obs[0])
                        print(f"[DEBUG]: fast_stop: measured initial speed (env 0) ≈ {speed0:.2f} m/s")
                except Exception:
                    pass


        with torch.inference_mode():
            mod = obs.clone()  # we will write desired quaternion (0:4) and body-frame pos offset (4:7)

            cur_positions_w = []
            cur_q_world = []
            goal_q_world = []

            # In case the initial shove didn't take (some setups overwrite at t=0), retry once on first loop.
            #if args_cli.trajectory_type == "fast_stop" and not fast_shove_done and step == 0 and (elapsed > args_cli.hold_time):
            #    fast_shove_done = _apply_fast_stop_initial_velocity(
            #        env, env_waypoints_world, speed_mps=args_cli.fast_stop_speed, label="retry@loop0"
            #    )

            for eid in range(num_envs):
                # -------------------
                # Extract current world pose from observation
                # -------------------
                e_env = obs[eid, 4:7].detach().cpu().numpy().astype(np.float64)
                q_cur = obs[eid, 7:11].detach().cpu().numpy().astype(np.float64)
                R_wb = quat_to_rotmat_wxyz(q_cur)
                origin_world = default_env_origins[eid]
                p_world = origin_world - (R_wb @ e_env)

                # Current target waypoint for this env
                wps = env_waypoints_world[eid]
                idx = int(cur_wp_idx[eid])
                target_world = wps[idx]

                # ------- Corner-aware look-ahead -------
                have_prev = idx > 0
                have_next = idx < (len(wps) - 1) or args_cli.loop
                corner_big = False
                next_heading_yaw = None
                lead_target_world = target_world.copy()

                if have_prev and have_next:
                    prev_wp = wps[idx - 1]
                    next_wp = wps[(idx + 1) % len(wps)] if args_cli.loop and idx == len(wps) - 1 else wps[idx + 1]
                    v_in = target_world - prev_wp
                    v_out = next_wp - target_world
                    corner_ang = angle_between(v_in[:2], v_out[:2])
                    corner_big = corner_ang >= corner_thresh
                    if corner_big:
                        # Heading for next leg:
                        next_heading_yaw = float(np.arctan2(v_out[1], v_out[0]))

                # Distance to current waypoint
                #vec_to_wp = target_world - p_world
                vec_to_wp = e_env
                dist_to_wp = float(np.linalg.norm(vec_to_wp))

                # Inside pre-turn zone?
                inside_pre = (dist_to_wp <= pre_r) and corner_big

                # ======================================================
                # Desired orientation q_goal (with FINAL-ORIENTATION LOCK)
                # ======================================================
                if orient_lock_active[eid]:
                    q_goal = orient_lock_q[eid].copy()
                elif alpha <= 0.0:
                    q_goal = q_cur.copy()  # hold during startup hold
                else:
                    if (idx == len(wps) - 1) and (args_cli.final_yaw_deg is not None) and (dist_to_wp < args_cli.wp_radius):
                        yaw_goal = float(np.deg2rad(args_cli.final_yaw_deg))
                    elif inside_pre and next_heading_yaw is not None:
                        yaw_goal = next_heading_yaw
                    else:
                        yaw_goal = float(np.arctan2(vec_to_wp[1], vec_to_wp[0])) if np.linalg.norm(vec_to_wp[:2]) > 1e-9 else float(quaternion_to_euler(q_cur)[2])
                    yaw_curr = float(quaternion_to_euler(q_cur)[2])
                    yaw_blend = (1.0 - alpha) * yaw_curr + alpha * yaw_goal
                    q_goal = euler_to_quaternion(0.0, 0.0, yaw_blend)

                # ------------------------------------------------------
                # Final-orientation lock activation 
                # ------------------------------------------------------
                if (
                    (args_cli.trajectory_type == "linear" or args_cli.trajectory_type == "fast_stop")
                    and idx == len(wps) - 1
                    and dist_to_wp <= float(args_cli.wp_radius)
                    and not orient_lock_active[eid]
                ):
                    if args_cli.final_yaw_deg is not None:
                        yaw_lock = float(np.deg2rad(args_cli.final_yaw_deg))
                    elif len(wps) >= 2:
                        v_last = wps[-1] - wps[-2]
                        yaw_lock = float(np.arctan2(v_last[1], v_last[0]))
                    else:
                        yaw_lock = float(quaternion_to_euler(q_cur)[2])
                    orient_lock_q[eid] = euler_to_quaternion(0.0, 0.0, yaw_lock)
                    orient_lock_active[eid] = True
                    print(f"[INFO]: Env {eid} final-orientation locked at {np.rad2deg(yaw_lock):.1f} deg.")
                    q_goal = orient_lock_q[eid].copy()

                # ==============================
                # Position goal (with corner lead)
                # ==============================
                if alpha <= 0.0:
                    e_new = np.zeros(3, dtype=np.float64)
                else:
                    if inside_pre and next_heading_yaw is not None:
                        prev_wp = wps[idx - 1]
                        next_wp = wps[(idx + 1) % len(wps)] if args_cli.loop and idx == len(wps) - 1 else wps[idx + 1]
                        v_out = unit(next_wp - target_world)
                        lead_target_world = target_world + v_out * lead_d
                    vec_goal = lead_target_world - p_world
                    e_traj = (R_wb.T @ vec_goal)  # body-frame offset

                    if inside_pre:
                        e_traj = e_traj * slow_scale

                    e_new = alpha * e_traj

                # Clip magnitude for safety
                lim = float(args_cli.offset_clip)
                nrm = float(np.linalg.norm(e_new))
                if lim > 0 and nrm > lim:
                    e_new = e_new * (lim / nrm)

                # Overwrite env goals in the observation tensor copy
                mod[eid, :4] = torch.tensor(q_goal, device=obs.device, dtype=obs.dtype)
                mod[eid, 4:7] = torch.tensor(e_new, device=obs.device, dtype=obs.dtype)

                # Orientation viz data
                cur_positions_w.append(p_world)
                cur_q_world.append(q_cur)
                goal_q_world.append(q_goal)

                # -------------------
                # Waypoint advancement logic (no dwell)
                # -------------------
                yaw_ok = True
                if yaw_tol > 0.0:
                    yaw_cur = float(quaternion_to_euler(q_cur)[2])
                    if inside_pre and next_heading_yaw is not None:
                        yaw_target_check = next_heading_yaw
                    else:
                        yaw_target_check = float(np.arctan2(vec_to_wp[1], vec_to_wp[0])) if np.linalg.norm(vec_to_wp[:2]) > 1e-9 else yaw_cur
                    dy = (yaw_target_check - yaw_cur + np.pi) % (2*np.pi) - np.pi
                    yaw_ok = abs(dy) <= yaw_tol

                if dist_to_wp <= float(args_cli.wp_radius) and yaw_ok:
                    if idx < len(wps) - 1:
                        cur_wp_idx[eid] = idx + 1
                    else:
                        if args_cli.loop and len(wps) > 1:
                            cur_wp_idx[eid] = 0
                            orient_lock_active[eid] = False  # unlock when looping starts over
                        else:
                            cur_wp_idx[eid] = idx  # stay at final

            # ===== Inference step =====
            actions = policy(mod)
            obs, rews, dones, infos = env.step(actions)

            # Update orientation viz
            if args_cli.viz_orient and len(cur_positions_w) == num_envs:
                orient_viz.update(
                    positions_world=np.asarray(cur_positions_w, dtype=np.float32),
                    q_current_world=np.asarray(cur_q_world, dtype=np.float32),
                    q_goal_world=np.asarray(goal_q_world, dtype=np.float32),
                )

            # ---- logging ----
            for eid in range(obs.shape[0]):
                idx = int(cur_wp_idx[eid])
                target_world = env_waypoints_world[eid][idx]

                e_env = obs[eid, 4:7].cpu().numpy()
                q_cur = obs[eid, 7:11].cpu().numpy()
                R_wb = quat_to_rotmat_wxyz(q_cur)
                origin_world = default_env_origins[eid]
                p_world = origin_world - (R_wb @ e_env)
                p_local = p_world - origin_world

                vec_world = target_world - p_world
                dist = float(np.linalg.norm(vec_world))
                yaw_to_wp = float(np.arctan2(vec_world[1], vec_world[0])) if np.linalg.norm(vec_world[:2]) > 1e-9 else float(quaternion_to_euler(q_cur)[2])
                q_t = euler_to_quaternion(0.0, 0.0, yaw_to_wp)
                oerr = float(quaternion_distance(q_cur, q_t))

                v_lin = obs[eid, 11:14].cpu().numpy()
                v_ang = obs[eid, 14:17].cpu().numpy()
                act = actions[eid].cpu().numpy()

                logs.append({
                    'step': step, 'env_id': eid, 'timestamp': time.time() - start_time,
                    'position': p_local, 'orientation': quaternion_to_euler(q_cur),
                    'velocity': v_lin, 'angular_velocity': v_ang,
                    'distance_to_goal': dist, 'distance_to_trajectory': dist, 'orientation_error': oerr,
                    'actions': act, 'wp_index': idx
                })

                with open(log_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([step, eid, time.time() - start_time,
                                *p_local, *quaternion_to_euler(q_cur),
                                dist, dist, oerr, *v_lin, *v_ang, *act, idx])

            step += 1

            if step % 100 == 0:
                d = []
                for eid in range(num_envs):
                    idx = int(cur_wp_idx[eid])
                    target_world = env_waypoints_world[eid][idx]
                    e_env = obs[eid, 4:7].cpu().numpy()
                    q_cur = obs[eid, 7:11].cpu().numpy()
                    R_wb = quat_to_rotmat_wxyz(q_cur)
                    origin_world = default_env_origins[eid]
                    p_world = origin_world - (R_wb @ e_env)
                    d.append(np.linalg.norm(target_world - p_world))
                print(f"[INFO]: step {step} | Avg Dist to Current WP: {np.mean(d):.3f} m")

            if dones[0]:
                obs, _ = env.reset()
                cur_wp_idx[:] = 0
                orient_lock_active[:] = False  # clear locks on reset
                if args_cli.trajectory_type == "fast_stop":
                    _apply_fast_stop_initial_velocity(env, env_waypoints_world, speed_mps=args_cli.fast_stop_speed, label="post-reset")
                    # verify
                    try:
                        if obs.shape[0] > 0:
                            speed0 = _estimate_world_speed_from_obs(obs[0])
                            print(f"[DEBUG]: fast_stop: measured speed after reset (env 0) ≈ {speed0:.2f} m/s")
                    except Exception:
                        pass
                print("[INFO]: Environment reset")

    # Save analysis
    print("[INFO]: Analyzing…")
    metrics = analyze_trajectory_following_performance(logs)
    with open(analysis_txt, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"[INFO]: Done. Results → {out_dir}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

