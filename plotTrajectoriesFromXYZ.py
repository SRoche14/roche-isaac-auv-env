#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def load_xyz_csv(path):
    """
    Load a CSV with header x,y,z and return Nx3 numpy array.
    """
    return np.loadtxt(path, delimiter=",", skiprows=1)

def main():
    # Filenames
    odom_file = "odomForward.csv"
    pos_file = "positionsForward.csv"
    pos_cfd_file = "positionsForwardCFD.csv"

    # --- Load data ---
    odom = load_xyz_csv(odom_file)             # shape (N1, 3)
    pos = load_xyz_csv(pos_file)               # shape (N2, 3)
    pos_cfd = load_xyz_csv(pos_cfd_file)       # shape (N3, 3)

    # --- Compute offset from first odom point ---
    # First row: (x0, y0, z0)
    offset = odom[0]    # shape (3,)
    print(f"Offset (first odom point): {offset}")

    Rz = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    # --- Align positionForward* trajectories ---
    pos_aligned = np.array(pos) @ Rz.T
    pos_cfd_aligned = np.array(pos_cfd) @ Rz.T

    pos_align_vector = offset - pos_aligned[0]
    pos_cfd_align_vector = offset - pos_cfd_aligned[0]

    pos_aligned += pos_align_vector
    pos_cfd_aligned += pos_cfd_align_vector

    # --- Plot 3D trajectories ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Odom trajectory
    ax.plot(odom[:, 0], odom[:, 1], odom[:, 2],
            label="odomLeft", linewidth=2)

    # PositionForward aligned
    ax.plot(pos_aligned[:, 0], pos_aligned[:, 1], pos_aligned[:, 2],
            label="positionLeftCuboid (aligned)", linewidth=2, linestyle="--")

    # PositionForwardCFD aligned
    ax.plot(pos_cfd_aligned[:, 0], pos_cfd_aligned[:, 1], pos_cfd_aligned[:, 2],
            label="positionLeftCFD (aligned)", linewidth=2, linestyle=":")

    # Mark starting point
    ax.scatter(odom[0, 0], odom[0, 1], odom[0, 2],
               s=50, marker="o", label="start (odom)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Forward Trajectories (aligned to odom start)")
    ax.legend()
    ax.grid(True)

    # Try to make axes roughly equal
    all_points = np.vstack([odom, pos_aligned, pos_cfd_aligned])
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
