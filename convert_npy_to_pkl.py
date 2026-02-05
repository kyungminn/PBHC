#!/usr/bin/env python
"""
Convert motion file to PBHC motion .pkl format (same save format for both).
- Input: .npy (inference motion) or .pkl (GMR format)
- Output: PBHC .pkl with joblib, nested {motion_name: motion_data}.

Usage:
    python convert_npy_to_pkl.py <input.npy|input.pkl> <output.pkl> [motion_name]
"""
import numpy as np
import joblib
import sys
import os
import pickle
import torch
from humanoidverse.isaac_utils.isaac_utils.rotations import quat_to_angle_axis


class NumpyUnpickler(pickle.Unpickler):
    """Custom unpickler for numpy 2.x compatibility."""
    def find_class(self, module, name):
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        elif module == 'numpy._core.umath':
            module = 'numpy.core.umath'
        return super().find_class(module, name)


def load_pkl_compat(filepath):
    """Load .pkl with joblib first, then pickle with numpy compat."""
    try:
        return joblib.load(filepath)
    except Exception:
        pass
    with open(filepath, 'rb') as f:
        return NumpyUnpickler(f).load()


def create_axis_onehot(seq_len, axis_list):
    """
    Create a (seq_len, len(axis_list), 3) tensor where each position has 1 at the specified axis index.
    
    Args:
        seq_len: sequence length
        axis_list: list of axis indices (0, 1, or 2) for each of the joints
    
    Returns:
        tensor of shape (seq_len, len(axis_list), 3) with one-hot encoding
    """
    num_joints = len(axis_list)
    result = np.zeros((seq_len, num_joints, 3), dtype=np.float32)
    for j, axis_idx in enumerate(axis_list):
        result[:, j, axis_idx] = 1
    return result


def create_pose_aa_29dof(root_ori_xyzw, dof_pos_29, dof_axis_list):
    """
    Create pose_aa (axis-angle representation) from root orientation and DOF positions.
    
    Args:
        root_ori_xyzw: (seq_len, 4) quaternion in XYZW format
        dof_pos_29: (seq_len, 29) DOF positions
        dof_axis_list: list of 29 axis indices (0=X, 1=Y, 2=Z) for each joint
    
    Returns:
        pose_aa: (seq_len, 33, 3) axis-angle representation
    """
    seq_len = root_ori_xyzw.shape[0]
    
    # Convert root quaternion (XYZW) to axis-angle
    angle, quat_axis = quat_to_angle_axis(torch.from_numpy(root_ori_xyzw).to(torch.float32))
    root_pose_aa = (angle.unsqueeze(1) * quat_axis).unsqueeze(1).numpy().astype(np.float32)
    
    # Convert DOF positions to axis-angle using one-hot encoding
    axis_onehot = create_axis_onehot(seq_len, dof_axis_list)
    dof_pose_aa = axis_onehot * dof_pos_29[:, :, None].astype(np.float32)
    
    # Last 3 joints are zeros (placeholder)
    last_pose_aa = np.zeros((seq_len, 3, 3), dtype=np.float32)
    
    return np.concatenate((root_pose_aa, dof_pose_aa, last_pose_aa), axis=1).astype(np.float32)


def convert_npy_to_pbhc_pkl(input_npy, output_pkl, motion_name=None):
    """
    Convert inference motion .npy file to PBHC motion .pkl format.
    
    Args:
        input_npy: path to input .npy file
        output_pkl: path to output .pkl file
        motion_name: name for the motion (optional, derived from filename if not provided)
    
    Returns:
        output_pkl: path to the generated PBHC motion file
    """
    # Generate motion name if not provided
    if motion_name is None:
        motion_name = os.path.splitext(os.path.basename(output_pkl))[0]
    
    print(f"Loading {input_npy}...")
    data = np.load(input_npy, allow_pickle=True).item()
    
    print(f"Data structure: {type(data)}")
    print(f"Data keys: {list(data.keys())}")
    
    # Print shapes for debugging
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Extract motion data
    # Typical inference motion structure has:
    # - root_pos or root_trans_offset: (num_frames, 3)
    # - root_rot or root_ori: (num_frames, 4) - quaternion
    # - dof_pos or dof: (num_frames, num_dofs)
    
    # Extract DOF positions
    if 'dof_pos' in data:
        dof_pos = data['dof_pos']
    elif 'dof' in data:
        dof_pos = data['dof']
    else:
        raise ValueError("No 'dof_pos' or 'dof' key found in data")
    
    print(f"\nDOF positions shape: {dof_pos.shape}")
    num_frames = dof_pos.shape[0]
    num_dofs = dof_pos.shape[1]
    
    # Extract root position
    if 'root_pos' in data:
        root_pos = data['root_pos']
    elif 'root_trans_offset' in data:
        root_pos = data['root_trans_offset']
    elif 'root_trans' in data:
        root_pos = data['root_trans']
    else:
        print("Warning: No root position found, creating zeros")
        root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    
    print(f"Root position shape: {root_pos.shape}")
    
    # Extract root rotation
    if 'root_rot' in data:
        root_rot = data['root_rot']
    elif 'root_ori' in data:
        root_rot = data['root_ori']
    else:
        print("Warning: No root rotation found, creating identity quaternions")
        root_rot = np.tile([0, 0, 0, 1], (num_frames, 1)).astype(np.float32)
    
    print(f"Root rotation shape: {root_rot.shape}")
    
    # Check quaternion format and convert to XYZW if needed
    if root_rot.shape[1] == 4:
        first_quat = root_rot[0]
        max_idx = np.argmax(np.abs(first_quat))
        
        if max_idx == 0 and np.abs(first_quat[0]) > 0.5:
            print("Detected WXYZ format, converting to XYZW")
            root_rot_xyzw = np.concatenate([
                root_rot[:, 1:4],  # XYZ
                root_rot[:, 0:1],  # W
            ], axis=1)
        else:
            print("Detected XYZW format (keeping as is)")
            root_rot_xyzw = root_rot
    else:
        raise ValueError(f"Unexpected root_rot shape: {root_rot.shape}")
    
    # Create new motion data dictionary
    new_motion_data = {}
    new_motion_data['root_trans_offset'] = root_pos.astype(np.float32)
    new_motion_data['root_rot'] = root_rot_xyzw.astype(np.float32)
    new_motion_data['dof'] = dof_pos.astype(np.float32)
    
    # Set fps (default to 30 if not in data)
    if 'fps' in data:
        new_motion_data['fps'] = data['fps']
    else:
        print("Warning: fps not found, using default 30")
        new_motion_data['fps'] = 30
    
    # Create pose_aa
    # Axis list for 29 DOF
    axis_list_29dof = [
        1, 0, 2, 1, 1, 0,  # Left leg: hip_yaw(Y), hip_roll(X), hip_pitch(Z), knee(Y), ankle_pitch(Y), ankle_roll(X)
        1, 0, 2, 1, 1, 0,  # Right leg: same as left
        2, 0, 1,           # Waist: yaw(Z), roll(X), pitch(Y)
        1, 0, 2, 1, 0, 1, 2,  # Left arm: shoulder_pitch(Y), shoulder_roll(X), shoulder_yaw(Z), elbow(Y), wrist_roll(X), wrist_pitch(Y), wrist_yaw(Z)
        1, 0, 2, 1, 0, 1, 2,  # Right arm: same as left
    ]
    
    # Use appropriate axis list based on number of DOFs
    if num_dofs == 29:
        axis_list = axis_list_29dof
        print("Using 29 DOF axis list")
    elif num_dofs == 23:
        # 23 DOF (no wrists)
        axis_list = [
            1, 0, 2, 1, 1, 0,  # Left leg
            1, 0, 2, 1, 1, 0,  # Right leg
            2, 0, 1,           # Waist
            1, 0, 2, 1,        # Left arm (no wrist)
            1, 0, 2, 1,        # Right arm (no wrist)
        ]
        print("Using 23 DOF axis list")
    else:
        raise ValueError(f"Unsupported number of DOFs: {num_dofs}")
    
    print(f"Computing pose_aa from root_rot and dof using axis-angle conversion")
    new_motion_data['pose_aa'] = create_pose_aa_29dof(
        root_rot_xyzw,
        dof_pos,
        axis_list
    )
    print(f"Created pose_aa with shape {new_motion_data['pose_aa'].shape}")
    
    # Create smpl_joints (required by motion_lib)
    print(f"Creating smpl_joints with shape ({num_frames}, 33, 3)")
    new_motion_data['smpl_joints'] = np.zeros((num_frames, 33, 3), dtype=np.float32)
    
    print(f"\nNew motion data keys: {list(new_motion_data.keys())}")
    print(f"  root_trans_offset: {new_motion_data['root_trans_offset'].shape}")
    print(f"  root_rot: {new_motion_data['root_rot'].shape}")
    print(f"  dof: {new_motion_data['dof'].shape}")
    print(f"  pose_aa: {new_motion_data['pose_aa'].shape}")
    print(f"  smpl_joints: {new_motion_data['smpl_joints'].shape}")
    print(f"  fps: {new_motion_data['fps']}")
    
    # Save in nested structure format
    save_data = {motion_name: new_motion_data}
    
    print(f"\nSaving to {output_pkl}...")
    joblib.dump(save_data, output_pkl)
    print("Done!")
    
    # Verify
    print("\nVerifying saved file...")
    verified = joblib.load(output_pkl)
    verified_motion = verified[motion_name]
    print(f"Verified dof shape: {verified_motion['dof'].shape}")
    print(f"Verified pose_aa shape: {verified_motion['pose_aa'].shape}")
    print(f"Verified fps: {verified_motion['fps']}")
    print(f"Motion keys: {list(verified_motion.keys())}")
    
    return output_pkl


# Axis list for 29 DOF (shared)
AXIS_LIST_29DOF = [
    1, 0, 2, 1, 1, 0,  # Left leg
    1, 0, 2, 1, 1, 0,  # Right leg
    2, 0, 1,           # Waist
    1, 0, 2, 1, 0, 1, 2,  # Left arm
    1, 0, 2, 1, 0, 1, 2,  # Right arm
]
AXIS_LIST_23DOF = [
    1, 0, 2, 1, 1, 0, 1, 0, 2, 1, 1, 0,
    2, 0, 1,
    1, 0, 2, 1, 1, 0, 2, 1,
]


def _normalize_root_rot_to_xyzw(root_rot):
    """Ensure quaternion is XYZW; convert from WXYZ if needed."""
    if root_rot.shape[1] != 4:
        raise ValueError(f"Unexpected root_rot shape: {root_rot.shape}")
    first_quat = root_rot[0]
    max_idx = np.argmax(np.abs(first_quat))
    if max_idx == 0 and np.abs(first_quat[0]) > 0.5:
        return np.concatenate([root_rot[:, 1:4], root_rot[:, 0:1]], axis=1).astype(np.float32)
    return root_rot.astype(np.float32)


def _build_pbhc_motion_and_save(root_pos, root_rot_xyzw, dof_pos, fps, motion_name, output_pkl):
    """Build PBHC motion dict and save as {motion_name: new_motion_data} with joblib."""
    num_frames = dof_pos.shape[0]
    num_dofs = dof_pos.shape[1]
    axis_list = AXIS_LIST_29DOF if num_dofs == 29 else AXIS_LIST_23DOF
    if num_dofs not in (23, 29):
        raise ValueError(f"Unsupported number of DOFs: {num_dofs}")
    new_motion_data = {
        'root_trans_offset': root_pos.astype(np.float32),
        'root_rot': root_rot_xyzw,
        'dof': dof_pos.astype(np.float32),
        'fps': int(fps) if isinstance(fps, (float, np.floating)) else fps,
        'pose_aa': create_pose_aa_29dof(root_rot_xyzw, dof_pos, axis_list),
        'smpl_joints': np.zeros((num_frames, 33, 3), dtype=np.float32),
    }
    save_data = {motion_name: new_motion_data}
    joblib.dump(save_data, output_pkl)
    return output_pkl


def convert_gmr_pkl_to_pbhc_pkl(input_pkl, output_pkl, motion_name=None):
    """
    Convert GMR format .pkl to PBHC motion .pkl (same save format as npy conversion).
    Handles nested {motion_name: data} or flat {dof_pos, root_rot, ...}.
    """
    if motion_name is None:
        motion_name = os.path.splitext(os.path.basename(output_pkl))[0]
    print(f"Loading GMR pkl: {input_pkl}...")
    data = load_pkl_compat(input_pkl)
    print(f"Data type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    # Nested: {motion_name: {dof_pos, root_rot, ...}}
    if isinstance(data, dict) and len(data) == 1:
        first_key = list(data.keys())[0]
        first_val = data[first_key]
        if isinstance(first_val, dict) and ('dof_pos' in first_val or 'dof' in first_val):
            motion_data = first_val
            if motion_name == os.path.splitext(os.path.basename(output_pkl))[0]:
                motion_name = first_key.replace(' ', '_')
        else:
            motion_data = data
    else:
        motion_data = data
    if 'dof_pos' in motion_data:
        dof_pos = np.asarray(motion_data['dof_pos'], dtype=np.float32)
    elif 'dof' in motion_data:
        dof_pos = np.asarray(motion_data['dof'], dtype=np.float32)
    else:
        raise ValueError("No 'dof_pos' or 'dof' in motion data")
    num_frames = dof_pos.shape[0]
    if 'root_pos' in motion_data:
        root_pos = np.asarray(motion_data['root_pos'], dtype=np.float32)
    elif 'root_trans_offset' in motion_data:
        root_pos = np.asarray(motion_data['root_trans_offset'], dtype=np.float32)
    elif 'root_trans' in motion_data:
        root_pos = np.asarray(motion_data['root_trans'], dtype=np.float32)
    else:
        root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    if 'root_rot' in motion_data:
        root_rot = np.asarray(motion_data['root_rot'], dtype=np.float32)
    else:
        root_rot = np.tile([0, 0, 0, 1], (num_frames, 1)).astype(np.float32)
    root_rot_xyzw = _normalize_root_rot_to_xyzw(root_rot)
    fps = motion_data.get('fps', 30)
    if isinstance(fps, np.ndarray):
        fps = int(fps.flat[0])
    print(f"Frames: {num_frames}, DOFs: {dof_pos.shape[1]}, fps: {fps}")
    print(f"Saving to {output_pkl}...")
    _build_pbhc_motion_and_save(root_pos, root_rot_xyzw, dof_pos, fps, motion_name, output_pkl)
    print("Done!")
    verified = joblib.load(output_pkl)
    vm = verified[motion_name]
    print(f"Verified dof: {vm['dof'].shape}, pose_aa: {vm['pose_aa'].shape}, fps: {vm['fps']}")
    return output_pkl


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_npy_to_pkl.py <input.npy|input.pkl> <output.pkl> [motion_name]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_pkl = sys.argv[2]
    motion_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    ext = os.path.splitext(input_path)[1].lower()
    try:
        if ext == '.pkl':
            output_path = convert_gmr_pkl_to_pbhc_pkl(input_path, output_pkl, motion_name)
        else:
            output_path = convert_npy_to_pbhc_pkl(input_path, output_pkl, motion_name)
        print(f"\n✅ Successfully converted {input_path} -> {output_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
