"""
Compute hydrodynamic forces and torques on a rigid body

Based on the descriptions of the MuJoCo hydrodynamic model: https://mujoco.readthedocs.io/en/3.0.1/computation/fluid.html

Authors: Ethan Fahnestock, Levi "Veevee" Cai (cail@mit.edu), Steven Roche (rochesh@mit.edu)
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np 
import torch 
import os
import joblib
import time
import torch.nn as nn
import os
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
  from isaaclab.utils.math import quat_conjugate, quat_inv, quat_apply, convert_quat
except ModuleNotFoundError:
  # Fallback quaternion helpers (w,x,y,z) to avoid IsaacLab dependency.
  def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-9)

  def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    q = torch.as_tensor(q)
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

  def quat_inv(q: torch.Tensor) -> torch.Tensor:
    q = torch.as_tensor(q)
    conj = quat_conjugate(q)
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True).clamp_min(1e-9)
    return conj / norm_sq

  def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = torch.as_tensor(q)
    v = torch.as_tensor(v)
    qn = _normalize_quat(q)
    qw = qn[..., :1]
    qv = qn[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)

  def convert_quat(q: torch.Tensor, to: str = "wxyz") -> torch.Tensor:
    q = torch.as_tensor(q)
    if to == "wxyz":
      return q
    if to == "xyzw":
      return torch.cat([q[..., 1:], q[..., :1]], dim=-1)
    raise ValueError(f"Unsupported target quaternion format: {to}")

import numpy as np, sys
# If we're on NumPy 1.x, provide an alias so unpickling can find numpy._core
if np.__version__.split('.', 1)[0] == '1':
    sys.modules.setdefault('numpy._core', np.core)

class TransformerRegressor(nn.Module):
  """
  Mirror the transient CFD transformer used during training so torch.load can unpickle it.
  """
  def __init__(
    self,
    input_size: int,
    output_size: int,
    seq_len: int,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    ff_dim: int = 256,
    dropout: float = 0.1,
  ):
    super().__init__()
    self.input_proj = nn.Linear(input_size, d_model)
    self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=ff_dim,
      dropout=dropout,
      batch_first=True,
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.head = nn.Sequential(
      nn.LayerNorm(d_model),
      nn.Linear(d_model, d_model),
      nn.ReLU(),
      nn.Linear(d_model, output_size),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.input_proj(x)
    x = x + self.pos_embed
    x = self.encoder(x)
    x = x[:, -1, :]
    return self.head(x)

# Ensure torch.load can resolve TransformerRegressor when models were saved from __main__.
if "__main__" in sys.modules and not hasattr(sys.modules["__main__"], "TransformerRegressor"):
  sys.modules["__main__"].TransformerRegressor = TransformerRegressor

@dataclass
class HydrodynamicForceModels:
  # def get_added_mass_cuboid(self, fluid_density=997.0, L = 0.686, W = 0.207, H = 0.3175):
  #     """
  #     Returns the added mass for each axis for a cuboid in water.
  #     """
  #     volume = 0.02529488465
  #     m_a_x = 0.2 * fluid_density * volume
  #     m_a_y = 0.8 * fluid_density * volume
  #     m_a_z = 0.8 * fluid_density * volume
  #     return np.array([m_a_x, m_a_y, m_a_z])


  num_envs: int 
  device: torch.device
  debug: bool = False
  # Use transformer models trained on transient CFD data when True.
  use_transient_models: bool = False
  transient_window_size: int = 20
  transient_include_time: bool = False  # Adds a normalized time feature if the model expects it.
  _steady_models_loaded: bool = field(default=False, init=False)
  _transient_models_loaded: bool = field(default=False, init=False)
  _linvel_history: Optional[np.ndarray] = field(default=None, init=False, repr=False)
  _angvel_history: Optional[np.ndarray] = field(default=None, init=False, repr=False)
  _steady_linear_model: Optional[nn.Module] = field(default=None, init=False, repr=False)
  _steady_angular_model: Optional[nn.Module] = field(default=None, init=False, repr=False)
  _steady_scaler_X_linear: Optional[StandardScaler] = field(default=None, init=False, repr=False)
  _steady_scaler_y_linear: Optional[StandardScaler] = field(default=None, init=False, repr=False)
  _steady_scaler_X_angular: Optional[StandardScaler] = field(default=None, init=False, repr=False)
  _steady_scaler_y_angular: Optional[StandardScaler] = field(default=None, init=False, repr=False)
  _transient_linear_model: Optional[nn.Module] = field(default=None, init=False, repr=False)
  _transient_angular_model: Optional[nn.Module] = field(default=None, init=False, repr=False)
  _transient_scaler_X_linear: Optional[StandardScaler] = field(default=None, init=False, repr=False)
  _transient_scaler_y_linear: Optional[StandardScaler] = field(default=None, init=False, repr=False)
  _transient_scaler_X_angular: Optional[StandardScaler] = field(default=None, init=False, repr=False)
  _transient_scaler_y_angular: Optional[StandardScaler] = field(default=None, init=False, repr=False)
    # buffers for added-mass acceleration estimate
  # _have_prev: bool = False
  # _prev_linvels_b: torch.Tensor = None

  # def calculate_added_mass_forces(
  #     self,
  #     root_linvels_b: torch.Tensor,
  #     dt: float,
  #     fluid_density: float = 997.0,
  #     L: float = 0.686,
  #     W: float = 0.207,
  #     H: float = 0.3175,
  # ) -> torch.Tensor:
  #   """
  #   Compute added-mass translational forces in body frame:
  #       F_added_b = -M_a * a_b
  #   using finite-difference acceleration a_b ≈ (v_t - v_{t-1}) / dt.

  #   Returns forces of shape (num_envs, 3). Torques are ignored here (A_rot = 0).
  #   """
  #   # First call: no previous velocity -> no added-mass force
  #   if (not self._have_prev) or (self._prev_linvels_b is None):
  #       self._prev_linvels_b = root_linvels_b.clone()
  #       self._have_prev = True
  #       return torch.zeros_like(root_linvels_b)

  #   # Finite-difference acceleration in body frame
  #   a_b = (root_linvels_b - self._prev_linvels_b) / dt

  #   # Update buffer
  #   self._prev_linvels_b = root_linvels_b.clone()

  #   # Added-mass diagonal (surge, sway, heave)
  #   m_a_np = self.get_added_mass_cuboid(fluid_density, L, W, H)   # (3,) numpy
  #   m_a = torch.as_tensor(m_a_np, device=self.device, dtype=root_linvels_b.dtype)  # (3,)
  #   m_a = m_a.view(1, 3).repeat(self.num_envs, 1)                 # (num_envs, 3)

  #   # F_added_b = -M_a * a_b (element-wise, since we use diagonal added-mass)
  #   F_added_b = - m_a * a_b

  #   return F_added_b




  base_dir = '/home/warp/isaacsim4.5/IsaacLab/roche-isaac-auv-env'
  linear_model_path = os.path.join(base_dir, 'saved_best_models', 'best_model_linear.pt')
  angular_model_path = os.path.join(base_dir, 'saved_best_models', 'best_model_angular.pt')
  linear_scaler_X_path = os.path.join(base_dir, 'saved_best_models', 'scaler_X_linear.pkl')
  linear_scaler_y_path = os.path.join(base_dir, 'saved_best_models', 'scaler_y_linear.pkl')
  angular_scaler_X_path = os.path.join(base_dir, 'saved_best_models', 'scaler_X_angular.pkl')
  angular_scaler_y_path = os.path.join(base_dir, 'saved_best_models', 'scaler_y_angular.pkl')
  transient_dir = os.path.join(base_dir, 'saved_best_models_transient')
  transient_linear_model_path = os.path.join(transient_dir, 'best_model_transformer_linear.pt')
  transient_angular_model_path = os.path.join(transient_dir, 'best_model_transformer_angular.pt')
  transient_scaler_X_linear_path = os.path.join(transient_dir, 'scaler_X_transformer_linear.pkl')
  transient_scaler_y_linear_path = os.path.join(transient_dir, 'scaler_y_transformer_linear.pkl')
  transient_scaler_X_angular_path = os.path.join(transient_dir, 'scaler_X_transformer_angular.pkl')
  transient_scaler_y_angular_path = os.path.join(transient_dir, 'scaler_y_transformer_angular.pkl')

  def calculate_buoyancy_forces(self,
                                root_quats_w: torch.tensor, # robot orientations in world frame
                                fluid_density: float, # fluid density
                                volumes: torch.tensor, # rigid body volume 
                                g_mag: float, # magnitude of gravity
                                com_to_cob_offsets:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Compute wrenches (forces and torques) due to buoyancy on fully-submerged rigid body in fluid.
    Returned forces and torques are in the body root frame.
    Note that gravity is applied by Isaac Sim by default.
    """

    buoyancy_forces_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
    buoyancy_torques_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

    buoyancy_directions_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
    buoyancy_directions_w[..., 2] = 1.0 # opposing gravity vector in the world frame
    
    if self.debug: print(f"shape of root_quats: {root_quats_w.shape}, shape of buoyancy_vectors: {buoyancy_directions_w.shape}")

    buoyancy_directions_b = quat_apply(quat_conjugate(root_quats_w), buoyancy_directions_w)

    # todo: we should actually be computing buoyancy forces at the root of the vehicle, not the COB, is this the same though?
    buoyancy_forces_at_cob_b = buoyancy_directions_b * fluid_density * volumes.repeat(1,3) * g_mag
    buoyancy_forces_b = buoyancy_forces_at_cob_b

    buoyancy_torques_b = torch.cross(com_to_cob_offsets, buoyancy_forces_at_cob_b, dim=-1) # torque = r x F

    if self.debug: print(f"Calculated buoyancy values: forces are {buoyancy_forces_b} and torques are {buoyancy_torques_b}")

    return (buoyancy_forces_b, buoyancy_torques_b)
  
  def _calculate_inferred_half_dimensions(self, inertias, masses):
    """
    Computes inferred half dimensions for an "equivalent inertia box" of the vehicle
    """
    r = torch.sqrt( (3/(2 * masses.repeat(1,3))) * (torch.roll(inertias, 1, 1) + torch.roll(inertias, -1, 1) - inertias))
    # r = torch.tensor([[0.869/2, 0.462/2, 0.1]], device=self.device).repeat(self.num_envs, 1)

    return r

  def _ensure_steady_models_loaded(self) -> None:
    """Load steady-state MLP models and scalers once, then reuse."""
    if self._steady_models_loaded:
      return
    self._steady_linear_model = torch.load(
      self.linear_model_path, weights_only=False, map_location=self.device
    ).to(self.device)
    self._steady_linear_model.eval()
    self._steady_angular_model = torch.load(
      self.angular_model_path, weights_only=False, map_location=self.device
    ).to(self.device)
    self._steady_angular_model.eval()
    self._steady_scaler_X_linear = joblib.load(self.linear_scaler_X_path)
    self._steady_scaler_y_linear = joblib.load(self.linear_scaler_y_path)
    self._steady_scaler_X_angular = joblib.load(self.angular_scaler_X_path)
    self._steady_scaler_y_angular = joblib.load(self.angular_scaler_y_path)
    self._steady_models_loaded = True

  def _ensure_transient_models_loaded(self) -> None:
    """Load transformer models and scalers once, then reuse."""
    if self._transient_models_loaded:
      return
    debug_load = self.debug or os.getenv("AUV_TRANSIENT_LOAD_DEBUG") == "1"
    load_device = self.device
    if getattr(load_device, "type", None) == "cuda" and not torch.cuda.is_available():
      if debug_load:
        print("[WARN] CUDA not available; loading transient models on CPU.", flush=True)
      load_device = torch.device("cpu")
    if debug_load:
      print(f"[INFO] Loading transient CFD transformer models on {load_device}...", flush=True)
    t0 = time.perf_counter()
    self._transient_linear_model = torch.load(
      self.transient_linear_model_path, weights_only=False, map_location=load_device
    )
    if debug_load:
      print(f"[INFO] transient linear torch.load took {time.perf_counter() - t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    self._transient_linear_model = self._transient_linear_model.to(self.device)
    if debug_load:
      print(f"[INFO] transient linear .to({self.device}) took {time.perf_counter() - t0:.2f}s", flush=True)
    self._transient_linear_model.eval()
    t0 = time.perf_counter()
    self._transient_angular_model = torch.load(
      self.transient_angular_model_path, weights_only=False, map_location=load_device
    )
    if debug_load:
      print(f"[INFO] transient angular torch.load took {time.perf_counter() - t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    self._transient_angular_model = self._transient_angular_model.to(self.device)
    if debug_load:
      print(f"[INFO] transient angular .to({self.device}) took {time.perf_counter() - t0:.2f}s", flush=True)
    self._transient_angular_model.eval()
    # Verify the window size matches the transformer positional embedding.
    if hasattr(self._transient_linear_model, "pos_embed"):
      seq_len = self._transient_linear_model.pos_embed.shape[1]
      if seq_len != self.transient_window_size:
        raise ValueError(
          f"transient_window_size={self.transient_window_size} "
          f"does not match transformer seq_len={seq_len}"
        )
    if hasattr(self._transient_angular_model, "pos_embed"):
      seq_len = self._transient_angular_model.pos_embed.shape[1]
      if seq_len != self.transient_window_size:
        raise ValueError(
          f"transient_window_size={self.transient_window_size} "
          f"does not match transformer seq_len={seq_len}"
        )
    if debug_load:
      print(f"[INFO] Loading scaler_X_linear from {self.transient_scaler_X_linear_path}...", flush=True)
    t0 = time.perf_counter()
    self._transient_scaler_X_linear = joblib.load(self.transient_scaler_X_linear_path)
    if debug_load:
      print(f"[INFO] scaler_X_linear load took {time.perf_counter() - t0:.2f}s", flush=True)
    if debug_load:
      print(f"[INFO] Loading scaler_y_linear from {self.transient_scaler_y_linear_path}...", flush=True)
    t0 = time.perf_counter()
    self._transient_scaler_y_linear = joblib.load(self.transient_scaler_y_linear_path)
    if debug_load:
      print(f"[INFO] scaler_y_linear load took {time.perf_counter() - t0:.2f}s", flush=True)
    if debug_load:
      print(f"[INFO] Loading scaler_X_angular from {self.transient_scaler_X_angular_path}...", flush=True)
    t0 = time.perf_counter()
    self._transient_scaler_X_angular = joblib.load(self.transient_scaler_X_angular_path)
    if debug_load:
      print(f"[INFO] scaler_X_angular load took {time.perf_counter() - t0:.2f}s", flush=True)
    if debug_load:
      print(f"[INFO] Loading scaler_y_angular from {self.transient_scaler_y_angular_path}...", flush=True)
    t0 = time.perf_counter()
    self._transient_scaler_y_angular = joblib.load(self.transient_scaler_y_angular_path)
    if debug_load:
      print(f"[INFO] scaler_y_angular load took {time.perf_counter() - t0:.2f}s", flush=True)
    self._transient_models_loaded = True

  def _update_transient_history(self, history: np.ndarray, new_vals: np.ndarray) -> np.ndarray:
    """Roll a (num_envs, window, 3) buffer and append latest values."""
    history[:, :-1, :] = history[:, 1:, :]
    history[:, -1, :] = new_vals
    return history

  def _ensure_transient_buffers(self, lin_np: np.ndarray, ang_np: np.ndarray) -> None:
    """Initialize or update rolling velocity buffers for transformer inputs."""
    if (
      self._linvel_history is None
      or self._angvel_history is None
      or self._linvel_history.shape[0] != lin_np.shape[0]
    ):
      self._linvel_history = np.repeat(lin_np[:, None, :], self.transient_window_size, axis=1)
      self._angvel_history = np.repeat(ang_np[:, None, :], self.transient_window_size, axis=1)
      return
    self._linvel_history = self._update_transient_history(self._linvel_history, lin_np)
    self._angvel_history = self._update_transient_history(self._angvel_history, ang_np)

  def _build_transient_features(self, seq: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Match the transformer training features: v_xyz, diff(v_xyz), optional time."""
    diff = np.zeros_like(seq)
    diff[:, 1:, :] = seq[:, 1:, :] - seq[:, :-1, :]
    features = np.concatenate([seq, diff], axis=2)
    expected = int(scaler.mean_.shape[0])
    want_time = self.transient_include_time or expected == features.shape[2] + 1
    if want_time:
      if expected != features.shape[2] + 1:
        raise ValueError(f"Expected {expected} features but time feature was requested.")
      t_norm = np.linspace(0.0, 1.0, seq.shape[1], dtype=np.float32)
      t_feat = np.broadcast_to(t_norm[None, :, None], (seq.shape[0], seq.shape[1], 1))
      features = np.concatenate([features, t_feat], axis=2)
    elif expected != features.shape[2]:
      raise ValueError(f"Unexpected feature count {expected} for transformer scaler.")
    return features.astype(np.float32, copy=False)

  def _predict_steady_drag_forces(
    self,
    root_linvels_b: torch.tensor,
    root_angvels_b: torch.tensor,
  ) -> Tuple[torch.tensor, torch.tensor]:
    """Predict forces/torques with steady-state MLP models."""
    self._ensure_steady_models_loaded()

    lin_np = root_linvels_b.detach().to('cpu').numpy()
    ang_np = root_angvels_b.detach().to('cpu').numpy()
    lin_std_np = self._steady_scaler_X_linear.transform(lin_np)
    ang_std_np = self._steady_scaler_X_angular.transform(ang_np)
    lin_std = torch.from_numpy(lin_std_np).to(self.device).type_as(root_linvels_b)
    ang_std = torch.from_numpy(ang_std_np).to(self.device).type_as(root_angvels_b)

    with torch.no_grad():
      pred_lin_std = self._steady_linear_model(lin_std)
      pred_ang_std = self._steady_angular_model(ang_std)

    pred_lin_std_np = pred_lin_std.detach().to('cpu').numpy()
    pred_ang_std_np = pred_ang_std.detach().to('cpu').numpy()
    pred_lin_np = self._steady_scaler_y_linear.inverse_transform(pred_lin_std_np)
    pred_ang_np = self._steady_scaler_y_angular.inverse_transform(pred_ang_std_np)

    pred_lin = torch.from_numpy(pred_lin_np).to(self.device).type_as(root_linvels_b)
    pred_ang = torch.from_numpy(pred_ang_np).to(self.device).type_as(root_angvels_b)

    forces_linear = pred_lin[:, :3] * 1000
    torques_linear = pred_lin[:, 3:] * 1000
    forces_angular = pred_ang[:, :3] * 1000
    torques_angular = pred_ang[:, 3:] * 1000

    forces = forces_linear + forces_angular
    torques = torques_linear + torques_angular
    return forces, torques

  def _predict_transient_drag_forces(
    self,
    root_linvels_b: torch.tensor,
    root_angvels_b: torch.tensor,
  ) -> Tuple[torch.tensor, torch.tensor]:
    """Predict forces/torques with transient transformer models using a rolling window."""
    self._ensure_transient_models_loaded()

    lin_np = root_linvels_b.detach().to('cpu').numpy().astype(np.float32)
    ang_np = root_angvels_b.detach().to('cpu').numpy().astype(np.float32)
    self._ensure_transient_buffers(lin_np, ang_np)

    X_lin = self._build_transient_features(self._linvel_history, self._transient_scaler_X_linear)
    X_ang = self._build_transient_features(self._angvel_history, self._transient_scaler_X_angular)
    X_lin_scaled = self._transient_scaler_X_linear.transform(X_lin.reshape(-1, X_lin.shape[2])).reshape(X_lin.shape)
    X_ang_scaled = self._transient_scaler_X_angular.transform(X_ang.reshape(-1, X_ang.shape[2])).reshape(X_ang.shape)

    lin_tensor = torch.from_numpy(X_lin_scaled).to(self.device).type_as(root_linvels_b)
    ang_tensor = torch.from_numpy(X_ang_scaled).to(self.device).type_as(root_angvels_b)

    with torch.no_grad():
      pred_lin_std = self._transient_linear_model(lin_tensor)
      pred_ang_std = self._transient_angular_model(ang_tensor)

    pred_lin_std_np = pred_lin_std.detach().to('cpu').numpy()
    pred_ang_std_np = pred_ang_std.detach().to('cpu').numpy()
    pred_lin_np = self._transient_scaler_y_linear.inverse_transform(pred_lin_std_np)
    pred_ang_np = self._transient_scaler_y_angular.inverse_transform(pred_ang_std_np)

    pred_lin = torch.from_numpy(pred_lin_np).to(self.device).type_as(root_linvels_b)
    pred_ang = torch.from_numpy(pred_ang_np).to(self.device).type_as(root_angvels_b)

    forces_linear = pred_lin[:, :3] * 1000
    torques_linear = pred_lin[:, 3:] * 1000
    forces_angular = pred_ang[:, :3] * 1000
    torques_angular = pred_ang[:, 3:] * 1000

    forces = forces_linear + forces_angular
    torques = torques_linear + torques_angular
    return forces, torques

  def _predict_steady_drag_components(
    self,
    root_linvels_b: torch.tensor,
    root_angvels_b: torch.tensor,
  ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """Predict linear/angular force/torque components with steady-state MLP models."""
    self._ensure_steady_models_loaded()

    lin_np = root_linvels_b.detach().to('cpu').numpy()
    ang_np = root_angvels_b.detach().to('cpu').numpy()
    lin_std_np = self._steady_scaler_X_linear.transform(lin_np)
    ang_std_np = self._steady_scaler_X_angular.transform(ang_np)
    lin_std = torch.from_numpy(lin_std_np).to(self.device).type_as(root_linvels_b)
    ang_std = torch.from_numpy(ang_std_np).to(self.device).type_as(root_angvels_b)

    with torch.no_grad():
      pred_lin_std = self._steady_linear_model(lin_std)
      pred_ang_std = self._steady_angular_model(ang_std)

    pred_lin_std_np = pred_lin_std.detach().to('cpu').numpy()
    pred_ang_std_np = pred_ang_std.detach().to('cpu').numpy()
    pred_lin_np = self._steady_scaler_y_linear.inverse_transform(pred_lin_std_np)
    pred_ang_np = self._steady_scaler_y_angular.inverse_transform(pred_ang_std_np)

    pred_lin = torch.from_numpy(pred_lin_np).to(self.device).type_as(root_linvels_b)
    pred_ang = torch.from_numpy(pred_ang_np).to(self.device).type_as(root_angvels_b)

    forces_linear = pred_lin[:, :3] * 1000
    torques_linear = pred_lin[:, 3:] * 1000
    forces_angular = pred_ang[:, :3] * 1000
    torques_angular = pred_ang[:, 3:] * 1000
    return forces_linear, torques_linear, forces_angular, torques_angular

  def _predict_transient_drag_components(
    self,
    root_linvels_b: torch.tensor,
    root_angvels_b: torch.tensor,
  ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """Predict linear/angular force/torque components with transient transformer models."""
    self._ensure_transient_models_loaded()

    lin_np = root_linvels_b.detach().to('cpu').numpy().astype(np.float32)
    ang_np = root_angvels_b.detach().to('cpu').numpy().astype(np.float32)
    self._ensure_transient_buffers(lin_np, ang_np)

    X_lin = self._build_transient_features(self._linvel_history, self._transient_scaler_X_linear)
    X_ang = self._build_transient_features(self._angvel_history, self._transient_scaler_X_angular)
    X_lin_scaled = self._transient_scaler_X_linear.transform(X_lin.reshape(-1, X_lin.shape[2])).reshape(X_lin.shape)
    X_ang_scaled = self._transient_scaler_X_angular.transform(X_ang.reshape(-1, X_ang.shape[2])).reshape(X_ang.shape)

    lin_tensor = torch.from_numpy(X_lin_scaled).to(self.device).type_as(root_linvels_b)
    ang_tensor = torch.from_numpy(X_ang_scaled).to(self.device).type_as(root_angvels_b)

    with torch.no_grad():
      pred_lin_std = self._transient_linear_model(lin_tensor)
      pred_ang_std = self._transient_angular_model(ang_tensor)

    pred_lin_std_np = pred_lin_std.detach().to('cpu').numpy()
    pred_ang_std_np = pred_ang_std.detach().to('cpu').numpy()
    pred_lin_np = self._transient_scaler_y_linear.inverse_transform(pred_lin_std_np)
    pred_ang_np = self._transient_scaler_y_angular.inverse_transform(pred_ang_std_np)

    pred_lin = torch.from_numpy(pred_lin_np).to(self.device).type_as(root_linvels_b)
    pred_ang = torch.from_numpy(pred_ang_np).to(self.device).type_as(root_angvels_b)

    forces_linear = pred_lin[:, :3] * 1000
    torques_linear = pred_lin[:, 3:] * 1000
    forces_angular = pred_ang[:, :3] * 1000
    torques_angular = pred_ang[:, 3:] * 1000
    return forces_linear, torques_linear, forces_angular, torques_angular

  def predict_drag_components(
    self,
    root_quats_w: torch.tensor,
    root_linvels_w: torch.tensor,
    root_angvels_w: torch.tensor,
  ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """Return linear/angular NN drag components along with body-frame velocities."""
    root_quats_b = quat_conjugate(root_quats_w)
    root_linvels_b = quat_apply(root_quats_b, root_linvels_w)
    root_angvels_b = quat_apply(root_quats_b, root_angvels_w)
    if self.use_transient_models:
      forces_linear, torques_linear, forces_angular, torques_angular = (
        self._predict_transient_drag_components(root_linvels_b, root_angvels_b)
      )
    else:
      forces_linear, torques_linear, forces_angular, torques_angular = (
        self._predict_steady_drag_components(root_linvels_b, root_angvels_b)
      )
    return forces_linear, torques_linear, forces_angular, torques_angular, root_linvels_b, root_angvels_b

  def calculate_quadratic_drag_forces(self,
                                  root_linvels_b: torch.tensor,
                                  root_angvels_b: torch.tensor,
                                  inertias: torch.tensor,
                                  masses: torch.tensor,
                                  fluid_density_rho
                                  ):
    # Choose steady-state (MLP) or transient (Transformer) models based on the flag.
    if self.use_transient_models:
      forces, torques = self._predict_transient_drag_forces(root_linvels_b, root_angvels_b)
    else:
      forces, torques = self._predict_steady_drag_forces(root_linvels_b, root_angvels_b)
    # The commented out code is how forces and torques were calculated prior to introducing the neural networks
    # ri = self._calculate_inferred_half_dimensions(inertias, masses)
    # rj = torch.roll(ri, 1, 1)
    # rk = torch.roll(ri, -1, 1)
    
    ri = self._calculate_inferred_half_dimensions(inertias, masses)
    rj = torch.roll(ri, 1, 1)
    rk = torch.roll(ri, -1, 1)

    forces = -2. * fluid_density_rho * rj * rk * torch.abs(root_linvels_b) * root_linvels_b
    torques = -0.5 * fluid_density_rho * ri * (torch.pow(rj,4) + torch.pow(rk,4)) * torch.abs(root_angvels_b) * root_angvels_b

    return (forces, torques)

  def get_total_mass_with_added(self, dry_mass):
    """
    Returns the total mass vector (dry + added) for each axis.
    """
    added_mass = self.get_added_mass_cuboid()
    return np.array([dry_mass, dry_mass, dry_mass]) + added_mass

  def calculate_linear_viscous_forces(self, 
                                      root_linvels_b: torch.tensor,
                                      root_angvels_b: torch.tensor,
                                      inertias: torch.tensor,
                                      masses,
                                      fluid_viscosity_beta
                                      ):
    ri = self._calculate_inferred_half_dimensions(inertias, masses)
    r_eq = torch.mean(ri, 1, keepdim=True)

    r_eq = r_eq.repeat(1,3)
    forces = -6. * fluid_viscosity_beta * torch.pi * r_eq * root_linvels_b
    torques = -8. * fluid_viscosity_beta * torch.pi * torch.pow(r_eq, 3) * root_angvels_b

    forces_zero = torch.zeros_like(forces)
    torques_zero = torch.zeros_like(torques)
    # return (forces_zero, torques_zero)
    return (forces, torques)

  def calculate_density_and_viscosity_forces(self, 
                                             root_quats_w: torch.tensor,
                                             root_linvels_w:torch.tensor, #[num_envs, 3]
                                             root_angvels_w:torch.tensor, #[num_envs, 3]
                                             inertias: torch.Tensor, #[num_envs, 3]
                                             inertias_mean: torch.Tensor, #[num_envs, 1]
                                             water_beta: float, 
                                             water_rho: float,
                                             masses: torch.tensor
                                             ):

    root_quats_b = quat_conjugate(root_quats_w)
    root_linvels_b = quat_apply(root_quats_b, root_linvels_w)
    root_angvels_b = quat_apply(root_quats_b, root_angvels_w)
  
    f_d, g_d = self.calculate_quadratic_drag_forces(root_linvels_b, root_angvels_b, inertias, masses, water_rho)
    f_v, g_v = self.calculate_linear_viscous_forces(root_linvels_b, root_angvels_b, inertias, masses, water_beta)
    return (f_d, g_d, f_v, g_v)

if __name__ == "__main__":
  # do some unit tests! 
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  water_rho = 997.0 # kg/m^3
  water_beta = 0.001306 # Pa s, dynamic viscosity of water @ 50 deg F
  g_mag = 9.81
  num_envs = 4
  com_to_cob_offset = torch.tensor([0.0, 0.0, 0.3], dtype=torch.float, device=device, requires_grad=False).reshape(1,3).repeat(num_envs, 1) 
  volume = 0.02529488465 # assuming cubic meters - NEUTRALLY BOUYANT

  # added new parameter to use transient vs. steady-state CFD models
  # use_transient_models = False sets to use steady-state MLP models
  forceModel = HydrodynamicForceModels(num_envs, device, debug=False, use_transient_models=True)

  root_quats = torch.tensor([[0.0, 0.0, 0.0, 1.0], # no rotation
                             [-0.7071068, 0, 0, 0.7071068], # 90 deg rotation about x
                             [ 0, -0.7071068, 0, 0.7071068 ], 
                             [ 0.3535534, 0.3535534, 0.1464466, 0.8535534 ], 
  
  ]).to(device)

  true_b_forces = torch.tensor([[0.0, 0.0, volume * water_rho * g_mag],
                                [0.0, -1 * volume * water_rho * g_mag, 0.0], 
                                [ volume * water_rho * g_mag, 0.0, 0.0], 
                                [ -0.5 * volume * water_rho * g_mag, 0.7071 * volume * water_rho * g_mag, 0.5 * volume * water_rho * g_mag], 
  
  ]).to(device)

  true_b_torques = torch.tensor([[0.0, 0.0, 0.0],
                                  [0.3 * water_rho * g_mag * volume, 0.0, 0.0],
                                  [0.0, 0.3 * water_rho * g_mag * volume, 0.0],
                                  [-0.3 * 0.7071 * water_rho * g_mag * volume, -0.15 * water_rho * g_mag * volume, 0.0],
  ]).to(device)

  b_force, b_torque = forceModel.calculate_buoyancy_forces( root_quats, water_rho, volume, g_mag, com_to_cob_offset)

  if(np.abs(b_force.cpu().numpy() - true_b_forces.cpu().numpy()).max() > 1e-9):
    print(f"ERROR: b_force is\n {b_force} \nand true_b_forces is \n {true_b_forces}\n with max value {np.abs(b_force.cpu().numpy() - true_b_forces.cpu().numpy()).max()}")
  if (np.abs(b_torque.cpu().numpy() - true_b_torques.cpu().numpy()).max() > 1e-9):
    print(f"ERROR: b_torque is\n {b_torque} and true_b_torques is\n {true_b_torques}")


  root_linvels = torch.tensor([[[0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0]],
  ])
  root_angvels = torch.tensor([[[0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0]],
  ])

  #env_inertia_tensors = 
  #dens_force, dense_torqe, visc_force, visc_torque = forceModel.calculate_density_and_viscosity_forces(root_linvels, root_angvels, env_inertia_tensors, water_beta, water_rho)
