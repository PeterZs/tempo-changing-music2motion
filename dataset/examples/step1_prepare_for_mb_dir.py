#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare BVH clips for MotionBuilder from a source directory.

This script:
1. Copies raw BVH files into <work_dir>/bvh (keeps filenames).
2. Inserts a T-pose frame at the beginning of each clip.
3. Writes results to <work_dir>/bvhForC for MotionBuilder retargeting.
"""

import argparse
import os
import shutil
import sys

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, repo_root)

from dataset.python import add_tpose_and_rename_clips


def run_prepare(src_dir: str, work_dir: str) -> None:
    src_dir = os.path.abspath(src_dir)
    work_dir = os.path.abspath(work_dir)

    if not os.path.isdir(src_dir):
        print(f"[Error] Source directory not found: {src_dir}")
        return

    bvh_files = add_tpose_and_rename_clips.list_bvh_files(src_dir)
    if not bvh_files:
        print(f"[Error] No .bvh files found in: {src_dir}")
        return

    dir_raw = os.path.join(work_dir, "bvh")
    dir_for_mb = os.path.join(work_dir, "bvhForC")
    os.makedirs(dir_raw, exist_ok=True)
    os.makedirs(dir_for_mb, exist_ok=True)

    copied_paths = []
    for fname in bvh_files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dir_raw, fname)
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copy(src_path, dst_path)
        copied_paths.append(dst_path)

    add_tpose_and_rename_clips.prepend_tpose_frame(copied_paths, dir_for_mb)

    print(f"Prepared {len(copied_paths)} BVH files.")
    print(f"Raw copies: {dir_raw}")
    print(f"Ready for MotionBuilder: {dir_for_mb}")
    print("\n[Next Step]")
    print("1. Open Autodesk MotionBuilder.")
    print("2. Run 'dataset/motionbuilder/PuppetToSmpl.py' (or a dataset-specific copy).")
    print(f"3. Ensure it points to: {dir_for_mb}")
    print("4. MotionBuilder will output to 'output' subfolder under bvhForC.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare BVH clips for MotionBuilder (keep filenames)."
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Directory containing BVH files.",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Output working directory.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Optional: only process one BVH filename (e.g. walk4_subject1.bvh).",
    )
    args = parser.parse_args()
    if args.only:
        src_dir = os.path.abspath(args.src)
        work_dir = os.path.abspath(args.work_dir)
        src_path = os.path.join(src_dir, args.only)
        if not os.path.exists(src_path):
            print(f"[Error] File not found: {src_path}")
            return
        dir_raw = os.path.join(work_dir, "bvh")
        dir_for_mb = os.path.join(work_dir, "bvhForC")
        os.makedirs(dir_raw, exist_ok=True)
        os.makedirs(dir_for_mb, exist_ok=True)
        dst_path = os.path.join(dir_raw, args.only)
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copy(src_path, dst_path)
        add_tpose_and_rename_clips.prepend_tpose_frame([dst_path], dir_for_mb)
        print(f"Prepared 1 BVH file: {args.only}")
        print(f"Ready for MotionBuilder: {dir_for_mb}")
    else:
        run_prepare(args.src, args.work_dir)


if __name__ == "__main__":
    main()

