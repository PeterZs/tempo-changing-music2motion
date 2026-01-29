#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to run the three proposed metrics:
1. Jo–Ha–Kyu (Tempo-motion synchronization)
2. S-curve (Motion aesthetics)
3. Head–Hand Contrast (Theatrical contrast)

This script loads example data (BVH/NPZ/TRC + WAV) and prints the scores.
"""

import os
import sys
import numpy as np

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, repo_root)

from metrics import jo_ha_kyu
from metrics import s_curve
from metrics import head_hand_contrast
from metrics import io_utils

def run_metrics(example_data_root):
    print(f"Running metrics on data from: {example_data_root}")
    
    # Paths to example files
    # Note: clip_001Re.bvh and clip_001Re.trc are the SMPL-retargeted versions
    bvh_path = os.path.join(example_data_root, "bvhSMPL", "clip_001Re.bvh")
    trc_path = os.path.join(example_data_root, "trcSMPL", "clip_001Re.trc")
    wav_path = os.path.join(example_data_root, "wav", "clip_001.wav")
    
    if not (os.path.exists(bvh_path) and os.path.exists(wav_path)):
        print("Error: Missing example files (BVH or WAV).")
        return

    # --- 1. Jo–Ha–Kyu Score ---
    print("\n--- [Metric 1] Jo–Ha–Kyu Score ---")
    print("Calculating correlation between music tempo and motion speed...")
    try:
        # Compute score from BVH
        result = jo_ha_kyu.compute_jo_ha_kyu_from_bvh_and_audio(
            bvh_path=bvh_path,
            audio_path=wav_path,
            fps_madmom=100  # Standard setting
        )
        print(f"Result: r = {result.r:.4f}, p-value = {result.p_value:.4e}")
        print(f" (Correlation > 0 indicates positive synchronization with tempo changes)")
    except Exception as e:
        print(f"Failed to compute Jo-Ha-Kyu: {e}")

    # --- 2. S-curve Score ---
    print("\n--- [Metric 2] S-curve Score ---")
    if os.path.exists(trc_path):
        print("Loading TRC for position-based metrics...")
        try:
            joints_dict, fps, _ = io_utils.read_trc(trc_path)
            
            # Extract Head and Hand positions
            # Joint names must match what is in the TRC file. 
            # Based on typical SMPL TRC exports: 'Head', 'Right_hand' (or 'RightHand')
            # Let's check keys or try standard names
            head_key = 'Head'
            hand_key = 'Right_hand' # Adjust based on your TRC column names
            
            # Fallback lookup if exact key not found
            keys = list(joints_dict.keys())
            if head_key not in keys:
                # Try to find something similar
                head_matches = [k for k in keys if 'head' in k.lower()]
                if head_matches: head_key = head_matches[0]
            
            if hand_key not in keys:
                hand_matches = [k for k in keys if 'right' in k.lower() and ('hand' in k.lower() or 'wrist' in k.lower())]
                if hand_matches: hand_key = hand_matches[0]
                
            # Also try to find Left Hand for comparison
            l_hand_key = 'Left_hand'
            if l_hand_key not in keys:
                l_hand_matches = [k for k in keys if 'left' in k.lower() and ('hand' in k.lower() or 'wrist' in k.lower())]
                if l_hand_matches: l_hand_key = l_hand_matches[0]

            print(f"Using joints: Head='{head_key}', R_Hand='{hand_key}', L_Hand='{l_hand_key}'")
            
            head_pos = joints_dict[head_key]
            r_hand_pos = joints_dict[hand_key]
            
            # Compute Head & Right Hand
            head_score, r_hand_score = s_curve.s_curve_scores_from_audio_and_positions(
                audio_path=wav_path,
                head_positions=head_pos,
                hand_positions=r_hand_pos,
                fps=fps
            )
            print(f"Head S-curve Score: {head_score:.4f}")
            print(f"Right Hand S-curve Score: {r_hand_score:.4f}")

            # Compute Left Hand if available
            if l_hand_key in joints_dict:
                l_hand_pos = joints_dict[l_hand_key]
                # We can reuse the single-joint function internally or just call the pair function again
                # Here we just call the core function for single joint
                # Note: io_utils doesn't export tempo_utils, we should use the imported module 'tempo_utils' directly?
                # Actually, we already imported 'from metrics import tempo_utils' at top level? No, we imported jo_ha_kyu etc.
                # Let's add the import if missing or use jo_ha_kyu's internal one, but better to import tempo_utils directly.
                # Checking imports... "from metrics import io_utils" is there. 
                # We need to import tempo_utils explicitly in this script.
                
                from metrics import tempo_utils as t_utils
                beat_times = t_utils.get_beat_times(wav_path, fps=100)
                l_hand_score = s_curve.s_curve_score_from_positions(l_hand_pos, beat_times, fps)
                print(f"Left Hand S-curve Score:  {l_hand_score:.4f}")
            
            print(" (Scores represent the percentage of beat segments with 'desirable' curvature)")
            
        except Exception as e:
            print(f"Failed to compute S-curve: {e}")
    else:
        print("Skipping S-curve (TRC file not found).")

    # --- 3. Head-Hand Contrast ---
    print("\n--- [Metric 3] Head-Hand Contrast Score ---")
    # Note: we renamed hand_pos to r_hand_pos above, so need to update here
    if os.path.exists(trc_path) and 'head_pos' in locals() and 'r_hand_pos' in locals():
        try:
            # Need parameters mu_x, sigma_x for the Gaussian mapping
            # These are typically learned from the training set.
            # We use placeholder values here or raw features.
            # Let's just print the raw features first.
            
            features = head_hand_contrast.head_hand_contrast_from_audio_and_positions(
                audio_path=wav_path,
                head_positions=head_pos,
                hand_positions=r_hand_pos,
                fps=fps
            )
            
            print(f"Raw Features:")
            print(f"  Xp Mean Diff: {features.xp_mean:.4f}")
            
            # To get a final score [0, 1], we would do:
            # score = head_hand_contrast.gaussian_contrast_score(features.xp_mean, mu_target, sigma_target)
            # print(f"Final Contrast Score: {score:.4f} (using dummy distribution)")
            
        except Exception as e:
            print(f"Failed to compute Contrast: {e}")
    else:
        print("Skipping Contrast (TRC file missing or load failed).")

if __name__ == "__main__":
    # Default example data path (relative to this script)
    # Updated to use the internal 'exampleData' inside the repo
    default_data = os.path.abspath(os.path.join(repo_root, "exampleData"))
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = default_data
        
    run_metrics(data_path)

