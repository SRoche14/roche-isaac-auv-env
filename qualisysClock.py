#!/usr/bin/env python3
"""
Locate the first run of k consecutive moving samples in Qualisys data (x/y by default).

Default input file: MITRE/Up/Curee Test0011.json
Sampling frequency: 100 Hz (Qualisys export)
Movement test: a sample is "moving" if either |dx| or |dy| exceeds the threshold.
The reported time corresponds to the first sample in the detected streak.
Marker names are matched flexibly when provided (e.g., "BlueROV - 1" resolves to "BlueROV2 - 1").
When no marker is specified, a CUREE marker is preferred; otherwise the most contiguous marker is selected.
If the path is under an "Up" directory, movement is detected on x/y/z.
"""

import argparse
import difflib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_PATH = Path("MITRE/Up/Curee Test0013.json")
DEFAULT_MARKER: Optional[str] = None
DEFAULT_FREQ = 100.0  # Hz
DEFAULT_K = 5
DEFAULT_THRESHOLD = 1e-3  # meters
PREFERRED_MARKER = "CUREE - 1"
FALLBACK_MARKER_PREFIX = "CUREE"


def _split_marker_label(label: str) -> Tuple[str, Optional[str]]:
    """Split a marker label into base and numeric suffix (e.g., 'BlueROV2', '1')."""
    base, sep, tail = label.partition("-")
    if not sep:
        return label.strip(), None
    tail_digits = tail.strip().replace(" ", "")
    if tail_digits.isdigit():
        return base.strip(), tail_digits
    return label.strip(), None


def _normalize_label(label: str) -> str:
    """Normalize labels for fuzzy matching (remove punctuation/spacing, lowercase)."""
    return "".join(ch.lower() for ch in label if ch.isalnum())


def resolve_marker_name(markers: List[Dict], requested: str) -> str:
    """Return the marker name from the file that best matches the requested label."""
    names = [m.get("Name") for m in markers if m.get("Name")]
    if not names:
        raise ValueError("No markers found in the Qualisys file")

    if requested in names:
        return requested

    lower_map = {n.lower(): n for n in names}
    if requested.lower() in lower_map:
        return lower_map[requested.lower()]

    req_base, req_idx = _split_marker_label(requested)
    prefix_matches = []
    for name in names:
        base, idx = _split_marker_label(name)
        if req_idx is not None:
            if idx == req_idx and base.lower().startswith(req_base.lower()):
                prefix_matches.append(name)
        elif base.lower().startswith(req_base.lower()):
            prefix_matches.append(name)

    if prefix_matches:
        return sorted(prefix_matches, key=len)[0]

    requested_norm = _normalize_label(requested)
    norm_map = {_normalize_label(n): n for n in names}
    close = difflib.get_close_matches(requested_norm, norm_map.keys(), n=1, cutoff=0.75)
    if close:
        return norm_map[close[0]]

    raise ValueError(f"Marker '{requested}' not found. Available markers: {names}")


def _marker_positions(marker: Dict) -> Optional[np.ndarray]:
    """Return xyz positions (m) for a single marker or None if it has no values."""
    name = marker.get("Name", "<unknown>")
    parts = marker.get("Parts", [])
    chunks = []
    for p in parts:
        vals = np.array(p.get("Values", []), dtype=float)
        if vals.size == 0:
            continue
        chunks.append(vals)
    if not chunks:
        return None
    vals = np.vstack(chunks)
    if vals.ndim != 2 or vals.shape[1] < 3:
        raise ValueError(f"Unexpected Values shape for '{name}': {vals.shape}")
    return vals[:, :3] / 1000.0  # mm -> m


def _marker_contiguous_lengths(marker: Dict) -> Tuple[int, int]:
    """Return (max contiguous length, total length) inferred from marker parts."""
    parts = marker.get("Parts", [])
    best = 0
    total = 0
    for part in parts:
        length = 0
        range_info = part.get("Range") or {}
        start = range_info.get("Start")
        end = range_info.get("End")
        if isinstance(start, int) and isinstance(end, int) and end >= start:
            length = end - start + 1
        else:
            values = part.get("Values", [])
            if isinstance(values, list):
                length = len(values)
        if length > best:
            best = length
        total += length
    return best, total


def _pick_best_marker(markers: List[Dict]) -> Optional[Dict]:
    candidates = []
    for marker in markers:
        best, total = _marker_contiguous_lengths(marker)
        if best <= 0:
            continue
        candidates.append((best, total, marker))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def load_positions(path: Path, marker_name: Optional[str]) -> np.ndarray:
    """Load concatenated xyz positions (m) for one marker or the best marker."""
    with path.open("r") as f:
        data = json.load(f)
    markers = data.get("Markers", [])
    if not markers:
        raise ValueError(f"No markers found in {path}")

    if not marker_name or str(marker_name).strip() in {"*", ""}:
        preferred = next((m for m in markers if m.get("Name") == PREFERRED_MARKER), None)
        if preferred is not None:
            best, _ = _marker_contiguous_lengths(preferred)
            if best > 0:
                chosen = preferred
            else:
                chosen = None
        else:
            chosen = None

        if chosen is None:
            curee_candidates = [
                marker
                for marker in markers
                if marker.get("Name", "").upper().startswith(FALLBACK_MARKER_PREFIX)
                and marker.get("Name") != PREFERRED_MARKER
            ]
            chosen = _pick_best_marker(curee_candidates)

        if chosen is None:
            chosen = _pick_best_marker(markers)

        if chosen is None:
            raise ValueError(f"No marker Values found in Parts in {path}")
        chosen_name = chosen.get("Name", "<unknown>")
        print(f"Using marker '{chosen_name}' (auto-selected)")
        vals = _marker_positions(chosen)
        if vals is None:
            raise ValueError(f"Marker '{chosen_name}' has no Values in Parts")
        return vals

    resolved = resolve_marker_name(markers, marker_name)
    if resolved != marker_name:
        print(f"Using marker '{resolved}' (requested '{marker_name}')")

    for marker in markers:
        if marker.get("Name") != resolved:
            continue
        vals = _marker_positions(marker)
        if vals is None:
            raise ValueError(f"Marker '{resolved}' has no Values in Parts")
        return vals
    raise ValueError(f"Marker '{resolved}' not found in {path}")


def find_first_consecutive_movement(
    positions: np.ndarray, k: int, threshold: float, axis: str = "xy"
) -> Optional[int]:
    """
    Return the index (in positions) of the first sample in the first streak of k moving samples.
    Movement is defined by |dx|>threshold or |dy|>threshold between successive samples,
    unless axis == "z" (|dz| only) or axis == "xyz" (any of x/y/z).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(positions) < k + 1:
        return None

    if axis == "z":
        diffs = np.diff(positions[:, 2], axis=0)
        moving = np.abs(diffs) > threshold
    elif axis == "xyz":
        diffs_xyz = np.diff(positions, axis=0)
        moving = np.any(np.abs(diffs_xyz) > threshold, axis=1)
    elif axis == "xy":
        diffs_xy = np.diff(positions[:, :2], axis=0)
        moving = np.any(np.abs(diffs_xy) > threshold, axis=1)
    else:
        raise ValueError(f"Unknown axis selection '{axis}'")

    streak = 0
    for i, is_moving in enumerate(moving):
        if is_moving:
            streak += 1
        else:
            streak = 0
        if streak >= k:
            # moving[i] refers to transition from sample i to i+1
            return i - k + 2  # index of first sample in the streak
    return None


def main():
    parser = argparse.ArgumentParser(description="Find first time of k consecutive movement samples.")
    parser.add_argument(
        "--path", type=Path, default=DEFAULT_PATH, help="Qualisys JSON file (default: MITRE run 1)"
    )
    parser.add_argument(
        "--marker",
        type=str,
        default=DEFAULT_MARKER,
        help="Marker name to read (default: marker with most contiguous values)",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Consecutive moving samples to detect")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Movement threshold in meters for axis deltas (x/y by default, z for Up)",
    )
    parser.add_argument(
        "--freq", type=float, default=DEFAULT_FREQ, help="Sampling frequency in Hz (default 100)"
    )
    args = parser.parse_args()

    positions = load_positions(args.path, args.marker)
    axis = "xyz" if "Up" in args.path.parts else "xy"
    idx = find_first_consecutive_movement(positions, args.k, args.threshold, axis=axis)

    if idx is None:
        print(f"No streak of {args.k} moving samples found in {len(positions)} samples.")
        return

    time_sec = idx / args.freq
    print(
        f"First streak of {args.k} moving samples starts at sample {idx} "
        f"(t = {time_sec:.3f} s) in file {args.path}"
    )


if __name__ == "__main__":
    main()
