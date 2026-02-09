#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare puppet BVH clips for MotionBuilder by:

1. Renaming them to a simple sequential pattern (clip_001.bvh, clip_002.bvh, ...),
2. Prepending a single T-pose frame to each clip.

This follows the workflow described in the JoruriPuppet appendix:
the IMU puppet skeleton BVH files are aligned to an SMPL T-pose so that
MotionBuilder can reliably create a Character and retarget to the SMPL rig.

Usage
-----
Example:

    python add_tpose_and_rename_clips.py \\
        --src path/to/bvhCut \\
        --dst-raw path/to/bvh \\
        --dst-for-mb path/to/bvhForC

The resulting folders correspond to:

- `bvh/`     – renamed raw BVH clips (no T-pose frame prepended),
- `bvhForC/` – BVH clips with a T-pose inserted as the first frame,
              to be imported into MotionBuilder for retargeting.
"""

import argparse
import os
import shutil
from typing import List

import numpy as np

# Ensure local helper modules are importable when run as a script
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bvh  # type: ignore


def list_bvh_files(src: str) -> List[str]:
    """List BVH files in a directory (sorted lexicographically)."""
    return sorted(
        [
            f
            for f in os.listdir(src)
            if os.path.isfile(os.path.join(src, f))
            and f.lower().endswith(".bvh")
            and not f.startswith("._")
        ]
    )


def _parse_joint_channels_from_header(header_lines: List[str]):
    """
    Parse BVH header lines and return a dict:
      joint_name -> {"start": int, "channels": List[str]}
    where "start" is the starting column in the motion vector for that joint.
    """
    joint_order: List[str] = []
    joint_channels: dict = {}
    current_joint = None

    for line in header_lines:
        s = line.strip()
        if s.startswith("ROOT "):
            parts = s.split()
            if len(parts) >= 2:
                current_joint = parts[1]
                joint_order.append(current_joint)
        elif s.startswith("JOINT "):
            parts = s.split()
            if len(parts) >= 2:
                current_joint = parts[1]
                joint_order.append(current_joint)
        elif s.startswith("CHANNELS "):
            if current_joint is None:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                n = int(parts[1])
            except Exception:
                continue
            joint_channels[current_joint] = parts[2 : 2 + n]

    col = 0
    joint_to_layout = {}
    for j in joint_order:
        ch = joint_channels.get(j, [])
        joint_to_layout[j] = {"start": col, "channels": ch}
        col += len(ch)
    return joint_to_layout


def copy_and_rename(src_dir: str, dst_raw_dir: str) -> List[str]:
    """Copy BVH files from src_dir into dst_raw_dir with sequential names.

    Returns
    -------
    dst_paths : list of str
        Absolute paths of the copied & renamed BVH files.
    """
    os.makedirs(dst_raw_dir, exist_ok=True)

    src_files = list_bvh_files(src_dir)
    dst_paths: List[str] = []

    for idx, fname in enumerate(src_files, start=1):
        new_name = f"clip_{idx:03d}.bvh"
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_raw_dir, new_name)
        
        # Avoid copying if source and destination are the same file
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copy(src_path, dst_path)
            
        dst_paths.append(dst_path)

    return dst_paths


def prepend_tpose_frame(bvh_paths: List[str], dst_for_mb_dir: str) -> None:
    """For each BVH, prepend a single T-pose frame and save to dst_for_mb_dir.

    The T-pose frame is constructed by:
    - preserving **translation channels** from the original first frame,
    - setting **rotation channels** to zero.

    If a reference T-pose BVH is found under the work dir (parent of dst_for_mb_dir),
    we will copy its per-joint **rotation channels** into the inserted frame by
    matching (joint_name, channel_name).

    Supported reference filenames (priority order):
    - T-pose.bvh
    - T-pose-*.bvh  (e.g., T-pose-bandai.bvh, T-pose-motorica.bvh)
    """
    os.makedirs(dst_for_mb_dir, exist_ok=True)

    # Optional reference T-pose
    tpose_ref_layout = None
    tpose_ref_frame0 = None
    tpose_ref_path_used = None
    force_zero_rotation_joints = set()
    try:
        work_dir = os.path.dirname(dst_for_mb_dir)
        candidates = [os.path.join(work_dir, "T-pose.bvh")]
        try:
            for fn in sorted(os.listdir(work_dir)):
                if fn.startswith("T-pose-") and fn.lower().endswith(".bvh"):
                    candidates.append(os.path.join(work_dir, fn))
        except Exception:
            pass

        for p in candidates:
            if os.path.exists(p):
                tpose_ref_path_used = p
                break

        if tpose_ref_path_used:
            ref_base = os.path.basename(tpose_ref_path_used).lower()
            if "bandai" in ref_base:
                # Bandai-specific user preference: keep both upper arms at zero.
                force_zero_rotation_joints = {"UpperArm_L", "UpperArm_R"}

            ref_data, _ref_fs, ref_header = bvh.bvhreader(tpose_ref_path_used)
            # bvhreader may return 1D array when BVH has exactly one frame.
            if getattr(ref_data, "ndim", None) == 1:
                ref_data = np.expand_dims(ref_data, axis=0)
            if ref_data.ndim == 2 and ref_data.shape[0] >= 1:
                tpose_ref_frame0 = ref_data[0].copy()
                tpose_ref_layout = _parse_joint_channels_from_header(ref_header)
    except Exception:
        tpose_ref_layout = None
        tpose_ref_frame0 = None
        tpose_ref_path_used = None
        force_zero_rotation_joints = set()

    for src_path in bvh_paths:
        data, fs, header = bvh.bvhreader(src_path)

        if data.ndim != 2 or data.shape[0] < 1:
            raise ValueError(f"Unexpected BVH data shape for {src_path}: {data.shape}")

        layout = _parse_joint_channels_from_header(header)
        first_frame = data[0].copy()
        tpose_1d = first_frame.copy()

        # Zero all rotation channels, preserve translations as-is.
        for _jname, info in layout.items():
            start = info["start"]
            channels = info["channels"]
            for ch in channels:
                if not ch.endswith("rotation"):
                    continue
                idx = start + channels.index(ch)
                if 0 <= idx < tpose_1d.shape[0]:
                    tpose_1d[idx] = 0.0

        # If we have a reference T-pose, copy its rotations by channel name.
        if tpose_ref_layout is not None and tpose_ref_frame0 is not None:
            d = tpose_1d.shape[0]
            for joint_name, dst_info in layout.items():
                if joint_name in force_zero_rotation_joints:
                    continue
                if joint_name not in tpose_ref_layout:
                    continue
                src_info = tpose_ref_layout[joint_name]

                dst_start = dst_info["start"]
                dst_channels = dst_info["channels"]
                src_start = src_info["start"]
                src_channels = src_info["channels"]

                for ch_name in dst_channels:
                    if not ch_name.endswith("rotation"):
                        continue
                    if ch_name not in src_channels:
                        continue
                    dst_idx = dst_start + dst_channels.index(ch_name)
                    src_idx = src_start + src_channels.index(ch_name)
                    if 0 <= dst_idx < d and 0 <= src_idx < tpose_ref_frame0.shape[0]:
                        tpose_1d[dst_idx] = float(tpose_ref_frame0[src_idx])

        tpose_frame = tpose_1d[None, :]

        data_new = np.concatenate([tpose_frame, data], axis=0)

        base_name = os.path.splitext(os.path.basename(src_path))[0]
        out_name = os.path.join(dst_for_mb_dir, base_name)
        bvh.bvhoutput(data_new, fs, out_name, header)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename puppet BVH clips and prepend a T-pose frame for "
            "MotionBuilder retargeting, following the JoruriPuppet workflow."
        )
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Source directory containing original BVH clips (e.g., 'bvhCut').",
    )
    parser.add_argument(
        "--dst-raw",
        required=True,
        help="Output directory for renamed raw BVH clips (e.g., 'bvh').",
    )
    parser.add_argument(
        "--dst-for-mb",
        required=True,
        help=(
            "Output directory for BVH clips with a prepended T-pose frame "
            "(e.g., 'bvhForC')."
        ),
    )

    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_raw_dir = os.path.abspath(args.dst_raw)
    dst_for_mb_dir = os.path.abspath(args.dst_for_mb)

    bvh_paths = copy_and_rename(src_dir, dst_raw_dir)
    prepend_tpose_frame(bvh_paths, dst_for_mb_dir)

    print(f"Copied and renamed {len(bvh_paths)} BVH files into: {dst_raw_dir}")
    print(
        f"Generated BVH files with prepended T-pose frames into: {dst_for_mb_dir}"
    )


if __name__ == "__main__":
    main()


