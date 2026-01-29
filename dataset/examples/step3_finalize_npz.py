#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 of the Dataset Pipeline: Finalize SMPL data (BVH -> NPZ).

This script assumes you have already:
1. Ran 'step1_prepare_for_mb.py'.
2. Ran 'dataset/motionbuilder/PuppetToSmpl.py' inside MotionBuilder.

It will:
1. Read the MotionBuilder output from 'exampleData/bvh_to_smpl_example/bvhForC/output'.
2. Standardize the SMPL BVH (remove T-pose, fix channels) into 'bvhSMPL'.
3. Convert to SMPL NPZ format (trans + poses) into 'npz'.
"""

import os
import sys

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, repo_root)

from dataset.python import smpl_bvh_to_smpl_npz

def run_step3_finalize():
    print(f"Running Step 3: Finalize SMPL data (BVH -> NPZ)")
    
    example_data_dir = os.path.join(repo_root, "exampleData")
    work_dir = os.path.join(example_data_dir, "bvh_to_smpl_example")
    
    # MotionBuilder output location (Input for this step)
    dir_mb_output = os.path.join(work_dir, "bvhForC", "output")
    
    # Outputs for this step
    dir_final_bvh = os.path.join(work_dir, "bvhSMPL")
    dir_final_npz = os.path.join(work_dir, "npz")
    
    # Check if input exists
    if not os.path.exists(dir_mb_output):
        print(f"[Error] MotionBuilder output directory not found:")
        print(f"  {dir_mb_output}")
        print("Did you run 'PuppetToSmpl.py' in MotionBuilder successfully?")
        return
        
    mb_files = [f for f in os.listdir(dir_mb_output) if f.endswith(".bvh")]
    if not mb_files:
        print(f"[Error] No .bvh files found in MotionBuilder output directory.")
        return

    print(f"Found {len(mb_files)} files from MotionBuilder.")

    # We need a reference SMPL T-pose BVH for the header.
    # It should be in dataset/motionbuilder/smpl-T.bvh
    ref_bvh_path = os.path.join(repo_root, "dataset", "motionbuilder", "smpl-T.bvh")
    
    if not os.path.exists(ref_bvh_path):
        print("[Warning] Reference 'smpl-T.bvh' not found in dataset/motionbuilder.")
        print("Using the first output file itself as header reference (fallback).")
        ref_bvh_path = os.path.join(dir_mb_output, mb_files[0])

    # --- Execution ---
    print(" -> Standardizing SMPL BVH (removing T-pose, fixing channels)...")
    smpl_bvh_to_smpl_npz.build_smpl_bvh(
        root=work_dir,
        smpl_t_bvh_path=ref_bvh_path,
        input_bvh_dir="bvhForC/output",  # Relative to work_dir
        out_bvh_smpl_dir="bvhSMPL",      # Relative to work_dir
    )
    
    print(" -> Converting to NPZ...")
    smpl_bvh_to_smpl_npz.build_smpl_npz(
        root=work_dir,
        in_bvh_smpl_dir="bvhSMPL",
        out_npz_dir="npz",
        # Convert root translation from centimeters to meters before saving NPZ.
        # If your BVH translation unit is not cm, adjust this (e.g., mm->m: 0.001).
        trans_scale=0.01,
    )
    
    print(f" -> SMPL NPZ generated in: {dir_final_npz}")
    if os.path.exists(dir_final_npz):
        print(f" -> Files: {os.listdir(dir_final_npz)}")
    print("\nDataset processing complete!")

if __name__ == "__main__":
    run_step3_finalize()

