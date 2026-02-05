#!/usr/bin/env python
"""
Convert 29 DOF motion data to 23 DOF by removing wrist joints.
Removes indices 19:22 (left wrist) and 26:29 (right wrist) from dof_pos and dof_vel.
Supports both pickle and joblib files, and nested PBHC format {motion_name: motion_data}.
"""
import pickle
import numpy as np
import sys

try:
    import joblib
except ImportError:
    joblib = None

# Handle numpy 2.0 compatibility with older Python/numpy
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect numpy._core to numpy.core
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        elif module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_pkl(filepath):
    """Load pkl file; try joblib first (PBHC format), then pickle."""
    if joblib is not None:
        try:
            data = joblib.load(filepath)
            return data, "joblib"
        except Exception:
            pass
    with open(filepath, 'rb') as f:
        data = NumpyUnpickler(f).load()
    return data, "pickle"


def convert_29_to_23(input_file, output_file):
    """Convert 29 DOF to 23 DOF by removing wrist joints."""
    print(f"Loading {input_file}...")
    data, load_fmt = load_pkl(input_file)
    print(f"Loaded with {load_fmt}")

    # Handle nested PBHC format: {motion_name: motion_data}
    motion_name = None
    if isinstance(data, dict) and len(data) == 1:
        key = list(data.keys())[0]
        val = data[key]
        if isinstance(val, dict) and ('dof' in val or 'dof_pos' in val):
            motion_name = key
            data = val
            print(f"Nested format detected, motion name: {motion_name}")

    print(f"Original keys: {list(data.keys())}")

    # Get DOF positions (PBHC uses 'dof', others use 'dof_pos')
    if 'dof_pos' in data:
        dof_key = 'dof_pos'
    elif 'dof' in data:
        dof_key = 'dof'
    else:
        raise ValueError("Neither 'dof_pos' nor 'dof' found in pkl file!")

    dof_pos = data[dof_key]
    if not isinstance(dof_pos, np.ndarray):
        dof_pos = np.array(dof_pos)
    print(f"Original {dof_key} shape: {dof_pos.shape}")

    # Remove indices 19:22 (left wrist) and 26:29 (right wrist). Keep [0:19, 22:26]
    if dof_pos.shape[-1] != 29:
        raise ValueError(f"Expected 29 DOF, got {dof_pos.shape[-1]}")
    dof_pos_23 = np.concatenate([
        dof_pos[..., :19],
        dof_pos[..., 22:26],
    ], axis=-1).astype(np.float32)
    print(f"New dof shape: {dof_pos_23.shape}")
    data[dof_key] = dof_pos_23

    # Convert dof_vel if present
    if 'dof_vel' in data:
        dof_vel = data['dof_vel']
        if not isinstance(dof_vel, np.ndarray):
            dof_vel = np.array(dof_vel)
        if dof_vel.shape[-1] == 29:
            data['dof_vel'] = np.concatenate([
                dof_vel[..., :19],
                dof_vel[..., 22:26],
            ], axis=-1).astype(np.float32)
            print(f"New dof_vel shape: {data['dof_vel'].shape}")

    # Convert pose_aa: 33 joints -> 27 (remove wrist joints at 20:23 and 26:29 in 1-indexed DOF)
    # pose_aa layout: root(0), joints 1-29 (DOF), padding 30-32. Remove 20:23 and 26:29.
    if 'pose_aa' in data:
        pa = data['pose_aa']
        if pa.shape[1] == 33:
            pa_23 = np.concatenate([
                pa[:, :20, :],   # root + joints 0:19
                pa[:, 23:27, :], # joints 23:26
                pa[:, 30:, :],   # padding
            ], axis=1).astype(np.float32)
            data['pose_aa'] = pa_23
            print(f"New pose_aa shape: {pa_23.shape}")

    # smpl_joints: keep as-is or reduce to 27 if needed for 23dof pipeline
    if 'num_dof' in data:
        data['num_dof'] = 23

    # Output: same structure as input
    to_save = data
    if motion_name is not None:
        to_save = {motion_name: data}

    print(f"\nSaving to {output_file}...")
    if load_fmt == "joblib" and joblib is not None:
        joblib.dump(to_save, output_file)
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(to_save, f)
    print("Done!")

    # Verify
    print("\nVerifying saved file...")
    verify, _ = load_pkl(output_file)
    if motion_name is not None:
        verify = verify[motion_name]
    dof_key_verify = 'dof' if 'dof' in verify else 'dof_pos'
    print(f"Verified {dof_key_verify} shape: {verify[dof_key_verify].shape}")
    if 'pose_aa' in verify:
        print(f"Verified pose_aa shape: {verify['pose_aa'].shape}")
    if 'dof_vel' in verify:
        print(f"Verified dof_vel shape: {verify['dof_vel'].shape}")

if __name__ == "__main__":
    input_file = "/home/kyungminlee/PBHC/baseball_batter.pkl"
    output_file = "/home/kyungminlee/PBHC/motion_data/baseball_batter_23dof.pkl"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    convert_29_to_23(input_file, output_file)
