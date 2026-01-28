#!/usr/bin/env python
"""
Convert 29 DOF motion data to 23 DOF by removing wrist joints.
Removes indices 19:22 (left wrist) and 26:29 (right wrist) from dof_pos and dof_vel.
"""
import pickle
import numpy as np
import sys

# Handle numpy 2.0 compatibility with older Python/numpy
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect numpy._core to numpy.core
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        elif module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

def convert_29_to_23(input_file, output_file):
    """Convert 29 DOF to 23 DOF by removing wrist joints."""
    print(f"Loading {input_file}...")
    with open(input_file, 'rb') as f:
        data = NumpyUnpickler(f).load()
    
    print(f"Original keys: {list(data.keys())}")
    
    # Check if dof_pos exists and has correct shape
    if 'dof_pos' not in data:
        raise ValueError("'dof_pos' not found in pkl file!")
    
    dof_pos = data['dof_pos']
    print(f"Original dof_pos shape: {dof_pos.shape}")
    
    # Convert to numpy if needed
    if not isinstance(dof_pos, np.ndarray):
        dof_pos = np.array(dof_pos)
    
    # Remove indices 19:22 and 26:29
    # Keep [0:19, 22:26, 29:]
    if dof_pos.shape[-1] == 29:
        dof_pos_23 = np.concatenate([
            dof_pos[..., :19],      # 0:19
            dof_pos[..., 22:26],    # 22:26 (skip 19:22)
        ], axis=-1)
        print(f"New dof_pos shape: {dof_pos_23.shape}")
    else:
        raise ValueError(f"Expected 29 DOF, got {dof_pos.shape[-1]}")
    
    # Update data dict
    data['dof_pos'] = dof_pos_23.astype(np.float32)
    
    # Handle dof_vel if it exists
    if 'dof_vel' in data:
        dof_vel = data['dof_vel']
        print(f"Original dof_vel shape: {dof_vel.shape}")
        
        if not isinstance(dof_vel, np.ndarray):
            dof_vel = np.array(dof_vel)
        
        if dof_vel.shape[-1] == 29:
            dof_vel_23 = np.concatenate([
                dof_vel[..., :19],
                dof_vel[..., 22:26],
            ], axis=-1)
            data['dof_vel'] = dof_vel_23.astype(np.float32)
            print(f"New dof_vel shape: {dof_vel_23.shape}")
    
    # Update num_dof if it exists
    if 'num_dof' in data:
        data['num_dof'] = 23
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print("Done!")
    
    # Verify
    print("\nVerifying saved file...")
    with open(output_file, 'rb') as f:
        data_verify = pickle.load(f)
    print(f"Verified dof_pos shape: {data_verify['dof_pos'].shape}")
    if 'dof_vel' in data_verify:
        print(f"Verified dof_vel shape: {data_verify['dof_vel'].shape}")

if __name__ == "__main__":
    input_file = "g1_ue_walk.pkl"
    output_file = "g1_ue_walk_23dof.pkl"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    convert_29_to_23(input_file, output_file)
