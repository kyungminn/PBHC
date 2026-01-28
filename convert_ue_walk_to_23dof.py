#!/usr/bin/env python
"""
Convert g1_ue_walk (29 DOF) to 23 DOF format matching the original processed format.
"""
import pickle
import numpy as np
import joblib
import sys
import torch
from humanoidverse.isaac_utils.isaac_utils.rotations import quat_to_angle_axis

# Handle numpy 2.0 compatibility
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        elif module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

def load_with_compat(filename):
    """Load pickle file with numpy 2.0 compatibility."""
    with open(filename, 'rb') as f:
        return NumpyUnpickler(f).load()

def create_axis_onehot(seq_len, axis_list):
    """
    Create a (seq_len, len(axis_list), 3) tensor where each position has 1 at the specified axis index.
    
    Args:
        seq_len: sequence length
        axis_list: list of axis indices (0, 1, or 2) for each of the 23 joints
    
    Returns:
        tensor of shape (seq_len, len(axis_list), 3) with one-hot encoding
    """
    num_joints = len(axis_list)
    result = np.zeros((seq_len, num_joints, 3), dtype=np.float32)
    for j, axis_idx in enumerate(axis_list):
        result[:, j, axis_idx] = 1
    return result

def create_pose_aa_23dof(root_ori_wxyz, dof_pos_23, dof_axis_list):
    """
    Create pose_aa (axis-angle representation) from root orientation and DOF positions.
    
    Args:
        root_ori_wxyz: (seq_len, 4) quaternion in WXYZ format
        dof_pos_23: (seq_len, 23) DOF positions
        dof_axis_list: list of 23 axis indices (0=X, 1=Y, 2=Z) for each joint
    
    Returns:
        pose_aa: (seq_len, 27, 3) axis-angle representation
    """
    seq_len = root_ori_wxyz.shape[0]
    
    # Convert root quaternion (WXYZ) to axis-angle
    angle, quat_axis = quat_to_angle_axis(torch.from_numpy(root_ori_wxyz).to(torch.float32))
    root_pose_aa = (angle.unsqueeze(1) * quat_axis).unsqueeze(1).numpy().astype(np.float32)
    
    # Convert DOF positions to axis-angle using one-hot encoding
    axis_onehot = create_axis_onehot(seq_len, dof_axis_list)
    dof_pose_aa = axis_onehot * dof_pos_23[:, :, None].astype(np.float32)
    
    # Last 3 joints are zeros (placeholder)
    last_pose_aa = np.zeros((seq_len, 3, 3), dtype=np.float32)
    
    return np.concatenate((root_pose_aa, dof_pose_aa, last_pose_aa), axis=1).astype(np.float32)

def convert_29_to_23(input_file, output_file):
    """Convert 29 DOF to 23 DOF in the correct format."""
    print(f"Loading {input_file}...")
    data_29 = load_with_compat(input_file)
    
    print(f"Original structure: {type(data_29)}")
    print(f"Top-level keys: {list(data_29.keys()) if isinstance(data_29, dict) else 'N/A'}")
    
    # Check if it's already nested or flat
    if isinstance(data_29, dict):
        # Check if it's a nested structure {motion_name: {motion_data}}
        first_key = next(iter(data_29))
        first_val = data_29[first_key]
        
        if isinstance(first_val, dict) and ('dof' in first_val or 'dof_pos' in first_val):
            # Nested structure
            motion_name = first_key
            motion_data = first_val
            print(f"Nested structure detected. Motion name: {motion_name}")
        elif 'dof_pos' in data_29 or 'dof' in data_29:
            # Flat structure - this is the motion data itself
            motion_data = data_29
            motion_name = 'g1_ue_walk'
            print(f"Flat structure detected. Using motion name: {motion_name}")
        else:
            raise ValueError(f"Cannot identify structure! Keys: {list(data_29.keys())}")
    else:
        raise ValueError("Unexpected data structure!")
    
    print(f"Motion data keys: {list(motion_data.keys())}")
    
    # Find the dof data
    if 'dof_pos' in motion_data:
        dof_key = 'dof_pos'
    elif 'dof' in motion_data:
        dof_key = 'dof'
    else:
        raise ValueError(f"Cannot find dof data! Keys: {list(motion_data.keys())}")
    
    dof_data = motion_data[dof_key]
    print(f"Original {dof_key} shape: {dof_data.shape}")
    
    if not isinstance(dof_data, np.ndarray):
        dof_data = np.array(dof_data)
    
    # Convert 29 -> 23 by removing indices 19:22 and 26:29
    if dof_data.shape[-1] == 29:
        dof_data_23 = np.concatenate([
            dof_data[..., :19],      # 0:19
            dof_data[..., 22:26],    # 22:26 (skip 19:22)
        ], axis=-1)
        print(f"New {dof_key} shape: {dof_data_23.shape}")
    else:
        raise ValueError(f"Expected 29 DOF, got {dof_data.shape[-1]}")
    
    # Create new motion data dict with proper keys matching the original format
    new_motion_data = {}
    num_frames = dof_data_23.shape[0]
    
    for key, val in motion_data.items():
        if key == dof_key:
            # Use 'dof' as the key (not 'dof_pos')
            new_motion_data['dof'] = dof_data_23.astype(np.float32)
        elif key == 'root_pos':
            # Rename to root_trans_offset
            new_motion_data['root_trans_offset'] = val
        elif key == 'root_rot':
            # Convert XYZW to WXYZ quaternion format
            if isinstance(val, np.ndarray):
                root_rot_wxyz = val[:, [3, 0, 1, 2]]  # XYZW -> WXYZ
                new_motion_data['root_rot'] = root_rot_wxyz.astype(np.float32)
                print(f"Converted root_rot from XYZW to WXYZ")
            else:
                new_motion_data['root_rot'] = val
        elif key in ['fps', 'pose_aa', 'smpl_joints']:
            # Keep these keys as is
            new_motion_data[key] = val
        # Skip other keys that don't match the expected format
    
    # Ensure fps is present (default to 30 if not)
    if 'fps' not in new_motion_data:
        print("Warning: fps not found, using default 30")
        new_motion_data['fps'] = 30
    
    # Create pose_aa properly (required by motion_lib)
    # 23 DOF axis list: each value indicates which axis (0=X, 1=Y, 2=Z) the joint rotates around
    axis_list_23dof = [
        1, 0, 2, 1, 1, 0,  # Left leg: hip_yaw(Y), hip_roll(X), hip_pitch(Z), knee(Y), ankle_pitch(Y), ankle_roll(X)
        1, 0, 2, 1, 1, 0,  # Right leg: same as left
        2, 0, 1,           # Waist: yaw(Z), roll(X), pitch(Y)
        1, 0, 2, 1,        # Left arm: shoulder_pitch(Y), shoulder_roll(X), shoulder_yaw(Z), elbow(Y)
        1, 0, 2, 1,        # Right arm: same as left
    ]
    
    if 'pose_aa' not in new_motion_data:
        print(f"Computing pose_aa from root_rot and dof using axis-angle conversion")
        if 'root_rot' in new_motion_data and 'dof' in new_motion_data:
            new_motion_data['pose_aa'] = create_pose_aa_23dof(
                new_motion_data['root_rot'],
                new_motion_data['dof'],
                axis_list_23dof
            )
            print(f"Created pose_aa with shape {new_motion_data['pose_aa'].shape}")
        else:
            print(f"Warning: Cannot compute pose_aa, creating zeros")
            new_motion_data['pose_aa'] = np.zeros((num_frames, 27, 3), dtype=np.float32)
    else:
        pose_aa = new_motion_data['pose_aa']
        print(f"Original pose_aa shape: {pose_aa.shape}")
    
    # Create smpl_joints if not exists (required by motion_lib)
    if 'smpl_joints' not in new_motion_data:
        print(f"Creating smpl_joints with shape ({num_frames}, 27, 3)")
        # Initialize with zeros - these will be computed by the motion library
        new_motion_data['smpl_joints'] = np.zeros((num_frames, 27, 3), dtype=np.float32)
    
    print(f"\nNew motion data keys: {list(new_motion_data.keys())}")
    
    # Wrap in top-level dict with motion name
    motion_name = 'g1_ue_walk'
    output_data = {motion_name: new_motion_data}
    
    # Save using joblib (same as original)
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        joblib.dump(output_data, f)
    print("Done!")
    
    # Verify
    print("\nVerifying saved file...")
    with open(output_file, 'rb') as f:
        verify_data = joblib.load(f)
    verify_motion = verify_data[motion_name]
    print(f"Verified dof shape: {verify_motion['dof'].shape}")
    print(f"Verified fps: {verify_motion.get('fps', 'N/A')}")
    print(f"Motion keys: {list(verify_motion.keys())}")

if __name__ == "__main__":
    input_file = "g1_ue_walk.pkl"
    output_file = "g1_ue_walk_23dof.pkl"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    convert_29_to_23(input_file, output_file)
