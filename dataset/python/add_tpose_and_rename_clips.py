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
            if os.path.isfile(os.path.join(src, f)) and f.lower().endswith(".bvh")
        ]
    )


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
    - copying the original first frame's Y translation (root height),
    - setting all other channels to zero.
    """
    os.makedirs(dst_for_mb_dir, exist_ok=True)

    for src_path in bvh_paths:
        data, fs, header = bvh.bvhreader(src_path)

        if data.ndim != 2 or data.shape[0] < 1:
            raise ValueError(f"Unexpected BVH data shape for {src_path}: {data.shape}")

        first_frame = data[0]
        tpose_frame = np.zeros_like(first_frame)[None, :]

        # Preserve root Y translation so the puppet stays at the same height.
        if first_frame.shape[0] >= 2:
            tpose_frame[0, 1] = first_frame[1]

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


