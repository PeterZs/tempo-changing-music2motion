#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert retargeted SMPL BVH files to:

1. Canonical SMPL BVH (using a reference SMPL T-pose BVH header),
2. SMPL `.npz` files with:
   - trans: root translation (T, 3)
   - poses: axis-angle joint rotations (T, J, 3), J = 24 (SMPL joints).

This script is a cleaned and parameterized version of the original
JoruriPuppet preprocessing code (`saveBVHsmpl_npz_2.py`), matching the
dataset structure described in the appendix:

- `bvhSMPL/` – SMPL BVH files,
- `npz/`     – SMPL NPZ files suitable for training.

Usage
-----
Example:

    python smpl_bvh_to_smpl_npz.py \\
        --root /path/to/sequence_root \\
        --smpl-t-bvh /path/to/smpl-T.bvh \\
        --input-bvh-dir bvhForC/output \\
        --out-bvh-smpl-dir bvhSMPL \\
        --out-npz-dir npz

The root directory then contains the three subfolders used in the
JoruriPuppet dataset (`bvhForC`, `bvhSMPL`, `npz`).
"""

import argparse
import os
from typing import List

import numpy as np

# Ensure local helper modules are importable when run as a script
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bvh  # type: ignore
import quat  # type: ignore


def get_bvh_files(directory: str) -> List[str]:
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.isfile(os.path.join(directory, f))
        and f.lower().endswith(".bvh")
    ]


def sample_every_three_2d(arr: np.ndarray) -> np.ndarray:
    """Downsample 6-channel per joint rotations to 3 channels per joint.

    The original MotionBuilder export uses 6 channels per joint
    (e.g., XYZ position + XYZ rotation). This function keeps every
    first 3 values in each 6-value block to obtain a [T, 3 * J] array
    of Euler angles.
    
    If the array already has 3 channels per joint (based on typical 
    SMPL joint count of 24), it returns the array as is.
    """
    if arr.shape[1] % 3 != 0:
        raise ValueError("Second dimension length must be a multiple of 3.")

    # If we have 72 columns (24 joints * 3 channels), assume it's already correct
    # (MotionBuilder 6-channel would be 144 columns for 24 joints)
    if arr.shape[1] == 72:
        return arr

    result = []
    for row in arr:
        sampled_row = []
        for i in range(0, len(row), 6):
            sampled_row.extend(row[i : i + 3])
        result.append(sampled_row)
    return np.array(result)


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


BVH_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
]


def build_smpl_bvh(
    root: str,
    smpl_t_bvh_path: str,
    input_bvh_dir: str = "bvhForC/output",
    out_bvh_smpl_dir: str = "bvhSMPL",
) -> None:
    """Create canonical SMPL BVH files using a reference SMPL T-pose BVH."""
    _, _, smpl_header = bvh.bvhreader(smpl_t_bvh_path)

    input_dir = os.path.join(root, input_bvh_dir)
    out_dir = os.path.join(root, out_bvh_smpl_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = get_bvh_files(input_dir)

    for item in files:
        filename = os.path.splitext(os.path.basename(item))[0]

        data, fs, _ = bvh.bvhreader(item)

        # Drop the first frame (T-pose inserted for retargeting)
        pos = data[1:, :3]
        rot_raw = data[1:, 3:]

        # Reduce from 6 channels per joint to 3 channels per joint
        rot_euler = sample_every_three_2d(rot_raw)

        data_new = np.concatenate([pos, rot_euler], axis=1)
        out_name = os.path.join(out_dir, filename)
        bvh.bvhoutput(data_new, fs, out_name, smpl_header)


def build_smpl_npz(
    root: str,
    in_bvh_smpl_dir: str = "bvhSMPL",
    out_npz_dir: str = "npz",
    trans_scale: float = 1.0,
) -> None:
    """Convert SMPL BVH files to SMPL-style NPZ files (trans + poses).

    Parameters
    ----------
    trans_scale : float
        Scale factor applied to BVH root translation before saving to NPZ.
        Use 0.01 for cm->m conversion (i.e., divide by 100).
    """
    in_dir = os.path.join(root, in_bvh_smpl_dir)
    out_dir = os.path.join(root, out_npz_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = get_bvh_files(in_dir)

    for item in files:
        filename = os.path.splitext(os.path.basename(item))[0]

        data, _, _ = bvh.bvhreader(item)

        pos = data[:, :3]
        if trans_scale != 1.0:
            pos = pos * float(trans_scale)
        rotation = data[:, 3:]

        rotation = rotation.reshape([rotation.shape[0], int(rotation.shape[1] / 3), 3])
        rotation = np.deg2rad(rotation)

        # Reorder joints to match SMPL joint order
        rotation_s = np.zeros_like(rotation)
        for i, joint in enumerate(SMPL_JOINT_NAMES):
            idx = BVH_JOINT_NAMES.index(joint)
            rotation_s[:, i] = rotation[:, idx]

        order = "zxy"
        quat_root_new = quat.from_euler(rotation_s, order=order)
        axis_angle = quat.to_axis_angle(quat_root_new)

        out_path = os.path.join(out_dir, filename + ".npz")
        np.savez(out_path, trans=pos, poses=axis_angle)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert MotionBuilder-retargeted SMPL BVH to canonical SMPL BVH "
            "and SMPL NPZ files, following the JoruriPuppet preprocessing "
            "pipeline."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory for one sequence (contains bvhForC/, etc.).",
    )
    parser.add_argument(
        "--smpl-t-bvh",
        required=True,
        help="Path to the SMPL T-pose BVH file (e.g., 'smpl-T.bvh').",
    )
    parser.add_argument(
        "--input-bvh-dir",
        default="bvhForC/output",
        help="Subdirectory under root containing retargeted BVH clips.",
    )
    parser.add_argument(
        "--out-bvh-smpl-dir",
        default="bvhSMPL",
        help="Subdirectory under root for canonical SMPL BVH output.",
    )
    parser.add_argument(
        "--out-npz-dir",
        default="npz",
        help="Subdirectory under root for SMPL NPZ output.",
    )
    parser.add_argument(
        "--trans-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to BVH root translation before saving to NPZ. "
            "Use 0.01 for cm->m conversion (divide by 100)."
        ),
    )

    args = parser.parse_args()

    root = os.path.abspath(args.root)
    smpl_t_bvh_path = os.path.abspath(args.smpl_t_bvh)

    build_smpl_bvh(
        root=root,
        smpl_t_bvh_path=smpl_t_bvh_path,
        input_bvh_dir=args.input_bvh_dir,
        out_bvh_smpl_dir=args.out_bvh_smpl_dir,
    )

    build_smpl_npz(
        root=root,
        in_bvh_smpl_dir=args.out_bvh_smpl_dir,
        out_npz_dir=args.out_npz_dir,
        trans_scale=args.trans_scale,
    )

    print(f"SMPL BVH written to: {os.path.join(root, args.out_bvh_smpl_dir)}")
    print(f"SMPL NPZ written to: {os.path.join(root, args.out_npz_dir)}")


if __name__ == "__main__":
    main()


