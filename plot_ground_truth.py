#!/usr/bin/env python3
"""
Plot ground-truth trajectory from a Qualisys Curee Test JSON file.

Usage:
  python3 plot_ground_truth.py --path MITRE/Up/Curee\\ Test0011.json
  python3 plot_ground_truth.py --path ... --marker "CUREE - 1"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _part_length(part: Dict) -> int:
    range_info = part.get("Range") or {}
    start = range_info.get("Start")
    end = range_info.get("End")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return end - start + 1
    values = part.get("Values", [])
    return len(values) if isinstance(values, list) else 0


def _contiguous_lengths(parts: List[Dict]) -> Tuple[int, int]:
    best = 0
    total = 0
    for part in parts:
        length = _part_length(part)
        if length > best:
            best = length
        total += length
    return best, total


def _best_entry(entries: List[Dict], name_prefix: Optional[str] = None) -> Optional[Dict]:
    candidates = []
    prefix = name_prefix.upper() if name_prefix else None
    for entry in entries:
        name = entry.get("Name", "")
        if prefix and not name.upper().startswith(prefix):
            continue
        best, total = _contiguous_lengths(entry.get("Parts", []))
        if best <= 0:
            continue
        candidates.append((best, total, entry))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _marker_positions(marker: Dict) -> Optional[np.ndarray]:
    name = marker.get("Name", "<unknown>")
    parts = marker.get("Parts", [])
    chunks = []
    for part in parts:
        vals = np.array(part.get("Values", []), dtype=float)
        if vals.size == 0:
            continue
        if vals.ndim != 2 or vals.shape[1] < 3:
            raise ValueError(f"Unexpected Values shape for '{name}': {vals.shape}")
        chunks.append(vals[:, :3])
    if not chunks:
        return None
    return np.vstack(chunks) / 1000.0  # mm -> m


def _rigid_body_sample_xyz(sample: object) -> Optional[List[float]]:
    if isinstance(sample, (list, tuple)) and sample:
        head = sample[0]
        if isinstance(head, (list, tuple)) and len(head) >= 3:
            return list(head[:3])
        if len(sample) >= 3 and all(isinstance(x, (int, float)) for x in sample[:3]):
            return list(sample[:3])
    return None


def _rigid_body_positions(rigid_body: Dict) -> Optional[np.ndarray]:
    parts = rigid_body.get("Parts", [])
    positions = []
    for part in parts:
        values = part.get("Values", [])
        if not isinstance(values, list):
            continue
        for sample in values:
            xyz = _rigid_body_sample_xyz(sample)
            if xyz is None:
                continue
            positions.append(xyz)
    if not positions:
        return None
    return np.array(positions, dtype=float) / 1000.0


def load_ground_truth(path: Path, marker_name: Optional[str]) -> Tuple[str, str, np.ndarray]:
    with path.open("r") as f:
        data = json.load(f)
    markers = data.get("Markers", [])
    rigid_bodies = data.get("RigidBodies", [])

    if marker_name:
        for marker in markers:
            if marker.get("Name") == marker_name:
                positions = _marker_positions(marker)
                if positions is None:
                    raise ValueError(f"Marker '{marker_name}' has no Values in Parts")
                return "marker", marker_name, positions
        raise ValueError(f"Marker '{marker_name}' not found in {path}")

    marker = _best_entry(markers, name_prefix="CUREE") or _best_entry(markers)
    if marker is not None:
        positions = _marker_positions(marker)
        if positions is not None:
            return "marker", marker.get("Name", "<unknown>"), positions

    rigid = _best_entry(rigid_bodies)
    if rigid is not None:
        positions = _rigid_body_positions(rigid)
        if positions is not None:
            return "rigid_body", rigid.get("Name", "<unknown>"), positions

    raise ValueError(f"No marker or rigid body data found in {path}")


def set_equal_axes(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    max_range = (maxs - mins).max()
    mids = 0.5 * (maxs + mins)
    ax.set_xlim(mids[0] - max_range / 2, mids[0] + max_range / 2)
    ax.set_ylim(mids[1] - max_range / 2, mids[1] + max_range / 2)
    ax.set_zlim(mids[2] - max_range / 2, mids[2] + max_range / 2)


def main():
    parser = argparse.ArgumentParser(description="Plot Qualisys ground-truth trajectory.")
    parser.add_argument("--path", type=Path, required=True, help="Path to Curee Test00XX.json")
    parser.add_argument("--marker", type=str, default=None, help="Marker name to use (optional)")
    args = parser.parse_args()

    source_type, source_name, positions = load_ground_truth(args.path, args.marker)
    print(f"Using {source_type} '{source_name}' with {len(positions)} samples.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="ground truth", linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], s=50, marker="o", label="start (GT)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"Ground Truth: {args.path.name}")
    ax.legend()
    ax.grid(True)
    set_equal_axes(ax, positions)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
