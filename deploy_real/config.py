import numpy as np
import yaml

class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            
            # action_scale can be a float or dict (per-joint)
            action_scale_raw = config["action_scale"]
            if isinstance(action_scale_raw, dict):
                # Per-joint action scale - need to map to dof_names order
                # G1 23DOF joint order:
                dof_names = [
                    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
                    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
                    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
                    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
                ]
                
                scales = []
                for name in dof_names:
                    # Extract core joint type from full name
                    if name.startswith("left_"):
                        core = name[5:]  # Remove "left_"
                    elif name.startswith("right_"):
                        core = name[6:]  # Remove "right_"
                    else:
                        core = name
                    
                    # Remove "_joint" suffix
                    if core.endswith("_joint"):
                        core = core[:-6]
                    
                    if core not in action_scale_raw:
                        raise KeyError(f"No action_scale found for joint: {name} (core={core})")
                    
                    scales.append(action_scale_raw[core])
                
                self.action_scale = np.array(scales, dtype=np.float32)
                print(f"Loaded per-joint action_scale: {self.action_scale}")
            else:
                # Fixed action scale for all joints
                self.action_scale = action_scale_raw
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.frame_stack = config.get("frame_stack", 5)
            self.motion_file = config["motion_file"]
            self.clip_action_limit = config["clip_action_limit"]
            
            # Student model parameters (optional, with defaults)
            self.future_max_steps = config.get("future_max_steps", 95)
            self.future_num_steps = config.get("future_num_steps", 20)
            self.history_length = config.get("history_length", 10)