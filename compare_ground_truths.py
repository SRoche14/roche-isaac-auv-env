#!/usr/bin/env python3
"""
Compare raw ground-truth trajectory vs RTS-smoothed ground truth.

Edit CUREE_JSON to point at the relative Curee Test00XX.json file,
or pass --path on the command line.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from plot_ground_truth import load_ground_truth, set_equal_axes
from plotMITRETrajectory import spike_smoothing

# Relative path to the Curee Test00XX.json file.
CUREE_JSON = Path("MITRE/Up/Curee Test0013.json")
GROUND_TRUTH_FREQ_HZ = 100.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw ground truth vs RTS-smoothed ground truth."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=CUREE_JSON,
        help="Relative path to Curee Test00XX.json",
    )
    parser.add_argument(
        "--marker",
        type=str,
        default=None,
        help="Marker name to use (optional)",
    )
    args = parser.parse_args()

    source_type, source_name, raw = load_ground_truth(args.path, args.marker)
    if len(raw) == 0:
        raise ValueError(f"No ground-truth samples found in {args.path}")

    smooth = spike_smoothing(raw, freq_hz=GROUND_TRUTH_FREQ_HZ)
    print(
        f"Using {source_type} '{source_name}' with {len(raw)} samples "
        f"(smoothed: {len(smooth)})."
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(raw[:, 0], raw[:, 1], raw[:, 2], label="raw ground truth", alpha=0.65)
    ax.plot(
        smooth[:, 0],
        smooth[:, 1],
        smooth[:, 2],
        label="RTS-smoothed ground truth",
        linewidth=2,
    )
    ax.scatter(raw[0, 0], raw[0, 1], raw[0, 2], s=50, marker="o", label="start")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"Raw vs RTS-smoothed GT: {args.path.name}")
    ax.legend()
    ax.grid(True)
    set_equal_axes(ax, np.vstack([raw, smooth]))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
