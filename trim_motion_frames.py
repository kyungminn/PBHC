#!/usr/bin/env python
"""Trim motion pkl files to first N frames and set fps."""
import joblib
import numpy as np
import sys

# Time-dimension keys (first axis = frames)
TIME_KEYS = (
    "root_trans_offset", "root_rot", "dof", "dof_pos", "dof_vel",
    "pose_aa", "smpl_joints",
)


def trim_and_save(filepath, num_frames=201, fps=50):
    """Load pkl, keep first num_frames, set fps, save in place."""
    print(f"Loading {filepath}...")
    data = joblib.load(filepath)

    # Nested format: {motion_name: motion_data}
    if isinstance(data, dict) and len(data) == 1:
        name = list(data.keys())[0]
        motion = data[name]
        if isinstance(motion, dict) and any(k in motion for k in ("dof", "dof_pos")):
            # Trim motion_data
            for key in TIME_KEYS:
                if key in motion and isinstance(motion[key], np.ndarray):
                    arr = motion[key]
                    if arr.shape[0] >= num_frames:
                        motion[key] = arr[:num_frames].copy()
                        print(f"  {key}: {arr.shape} -> {motion[key].shape}")
            motion["fps"] = fps
            joblib.dump(data, filepath)
            print(f"Saved first {num_frames} frames, fps={fps} -> {filepath}")
            return
    # Flat format
    motion = data
    for key in TIME_KEYS:
        if key in motion and isinstance(motion[key], np.ndarray):
            arr = motion[key]
            if arr.shape[0] >= num_frames:
                motion[key] = arr[:num_frames].copy()
                print(f"  {key}: {arr.shape} -> {motion[key].shape}")
    if "fps" in motion:
        motion["fps"] = fps
    joblib.dump(data, filepath)
    print(f"Saved first {num_frames} frames, fps={fps} -> {filepath}")


if __name__ == "__main__":
    num_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 201
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    files = [
        "/home/kyungminlee/PBHC/motion_data/walk_45cms_grounded_motion_23dof.pkl",
        "/home/kyungminlee/PBHC/motion_data/walk_45cms_grounded_motion.pkl",
    ]
    for path in files:
        trim_and_save(path, num_frames=num_frames, fps=fps)
        print()
