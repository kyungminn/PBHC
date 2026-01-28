from typing import Union
import numpy as np
import time
import torch
import joblib
import os
from datetime import datetime

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import (
        create_damping_cmd, 
        create_zero_cmd, 
        init_cmd_hg, 
        init_cmd_go,  
        MotorMode
    )
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from common.motion_lib_helper import get_motion_len
from config import Config
from collections import deque
import onnxruntime as ort
from omegaconf import DictConfig

# Import motion library for computing key body positions
from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX as MotionLibRobot


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q (w, x, y, z format)."""
    q_w, q_x, q_y, q_z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    t = 2.0 * np.cross(np.stack([q_x, q_y, q_z], axis=-1), v)
    return v - q_w[..., None] * t + np.cross(np.stack([q_x, q_y, q_z], axis=-1), t)


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.frame_stack = self.config.frame_stack
        
        self.motion_file = self.config.motion_file
        self.motion_len = get_motion_len(self.motion_file)
        
        # Load full motion data for student model
        self._load_motion_data()

        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = ort.InferenceSession(config.policy_path)
        
        # Check if this is a student model (requires future_motion_targets and prop_history)
        self.input_names = [inp.name for inp in self.policy.get_inputs()]
        self.is_student_model = "future_motion_targets" in self.input_names
        if self.is_student_model:
            print(f"Detected student model with inputs: {self.input_names}")
            # Student model parameters
            self.future_max_steps = getattr(config, 'future_max_steps', 95)
            self.future_num_steps = getattr(config, 'future_num_steps', 20)
            self.history_length = getattr(config, 'history_length', 10)
        else:
            print(f"Detected teacher model with inputs: {self.input_names}")
        
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.counter = 0

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.start_time = time.time()

        # Initialize histories for each observation type (for teacher model)
        self.history = {
            "action": deque(maxlen=self.frame_stack-1),
            "omega": deque(maxlen=self.frame_stack-1),
            "qj": deque(maxlen=self.frame_stack-1),
            "dqj": deque(maxlen=self.frame_stack-1),
            "gravity_orientation": deque(maxlen=self.frame_stack-1),
            "ref_motion_phase": deque(maxlen=self.frame_stack-1),
        }

        for _ in range(self.frame_stack - 1):
            for key in self.history:
                if key in ["action", "qj", "dqj"]:
                    self.history[key].append(torch.zeros(1, self.config.num_actions, dtype=torch.float))
                elif key in ["omega", "gravity_orientation"]:
                    self.history[key].append(torch.zeros(1, 3, dtype=torch.float))
                elif key == "ref_motion_phase":
                    self.history[key].append(torch.zeros(1, 1, dtype=torch.float))
                else:
                    raise ValueError(f"Not Implement: {key}")
        
        # Initialize histories for student model (prop_history)
        if self.is_student_model:
            self.student_history = {
                "base_ang_vel": deque(maxlen=self.history_length),
                "roll_pitch": deque(maxlen=self.history_length),
                "dof_pos": deque(maxlen=self.history_length),
                "dof_vel": deque(maxlen=self.history_length),
                "actions": deque(maxlen=self.history_length),
            }
            for _ in range(self.history_length):
                self.student_history["base_ang_vel"].append(np.zeros(3, dtype=np.float32))
                self.student_history["roll_pitch"].append(np.zeros(2, dtype=np.float32))
                self.student_history["dof_pos"].append(np.zeros(self.config.num_actions, dtype=np.float32))
                self.student_history["dof_vel"].append(np.zeros(self.config.num_actions, dtype=np.float32))
                self.student_history["actions"].append(np.zeros(self.config.num_actions, dtype=np.float32))
        
        # Observation logging setup
        self.obs_logging_enabled = getattr(config, 'obs_logging', False)
        self.obs_log_interval = getattr(config, 'obs_log_interval', 10)  # Log every N steps
        self.obs_log_data = []
        if self.obs_logging_enabled:
            self.obs_log_dir = os.path.join("logs", "obs_logs")
            os.makedirs(self.obs_log_dir, exist_ok=True)
            self.obs_log_file = os.path.join(
                self.obs_log_dir, 
                f"obs_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            print(f"Observation logging enabled. Will save to: {self.obs_log_file}")

    def _load_motion_data(self):
        """Load full motion data for future motion targets computation using MotionLib."""
        # Key body indices (same as urcirobot.py)
        self.key_body_id = [4, 6, 10, 12, 19, 23, 24, 25, 26]
        self.anchor_index = 0  # root
        
        # Load motion library for accurate body position computation
        motion_lib_cfg = DictConfig({
            'motion_file': self.motion_file,
            'asset': {
                'assetRoot': 'description/robots/g1/',
                'assetFileName': 'g1_23dof_lock_wrist_fitmotionONLY.xml',
            },
            'extend_config': [
                {'joint_name': 'left_hand_link', 'parent_name': 'left_elbow_link', 
                 'pos': [0.25, 0.0, 0.0], 'rot': [1.0, 0.0, 0.0, 0.0]},
                {'joint_name': 'right_hand_link', 'parent_name': 'right_elbow_link',
                 'pos': [0.25, 0.0, 0.0], 'rot': [1.0, 0.0, 0.0, 0.0]},
                {'joint_name': 'head_link', 'parent_name': 'torso_link',
                 'pos': [0.0, 0.0, 0.35], 'rot': [1.0, 0.0, 0.0, 0.0]},
            ],
        })
        
        self.motion_lib = MotionLibRobot(motion_lib_cfg, num_envs=1, device='cpu')
        self.motion_lib.load_motions(random_sample=False)
        
        # Get motion parameters from library
        self.motion_fps = int(1.0 / self.motion_lib._motion_dt[0].item())
        self.motion_dt = self.motion_lib._motion_dt[0].item()
        self.motion_num_frames = self.motion_lib._motion_num_frames[0].item()
        
        # Also load raw motion data for backward compatibility
        motion_data = joblib.load(self.motion_file)
        key = list(motion_data.keys())[0]
        data = motion_data[key]
        
        # Store motion data
        self.motion_root_pos = data["root_trans_offset"]  # (num_frames, 3)
        # Motion file stores quaternion in XYZW format, convert to WXYZ for internal use
        motion_rot_xyzw = data["root_rot"]  # (num_frames, 4) - quaternion (x, y, z, w)
        self.motion_root_rot = motion_rot_xyzw[:, [3, 0, 1, 2]]  # Convert XYZW to WXYZ
        self.motion_dof_pos = data["dof"]  # (num_frames, num_dofs)
        
        # Compute velocities from position differences
        self.motion_root_vel = np.zeros_like(self.motion_root_pos)
        self.motion_root_vel[1:] = (self.motion_root_pos[1:] - self.motion_root_pos[:-1]) / self.motion_dt
        
        # Compute angular velocities (simplified - from quaternion differences)
        self.motion_root_ang_vel = np.zeros((self.motion_num_frames, 3), dtype=np.float32)
        
        print(f"Loaded motion data with MotionLib: {self.motion_num_frames} frames at {self.motion_fps} fps")
    
    def _log_observation(self, actor_obs, future_motion_targets, prop_history, raw_obs):
        """Log observation data for debugging."""
        log_entry = {
            "counter": self.counter,
            "time": time.time(),
            "actor_obs": actor_obs.copy(),
            "future_motion_targets": future_motion_targets.copy(),
            "prop_history": prop_history.copy(),
            "action_output": self.action.copy(),
            "raw_obs": {k: v.copy() if hasattr(v, 'copy') else v for k, v in raw_obs.items()},
        }
        self.obs_log_data.append(log_entry)
        
        # Print summary every log interval
        if self.counter % (self.obs_log_interval * 10) == 0:
            print(f"[Step {self.counter}] Obs stats - "
                  f"actor_obs: mean={actor_obs.mean():.4f}, std={actor_obs.std():.4f}, "
                  f"action: mean={self.action.mean():.4f}, std={self.action.std():.4f}")
    
    def save_obs_log(self):
        """Save observation log to file."""
        if not self.obs_logging_enabled or len(self.obs_log_data) == 0:
            print("No observation data to save.")
            return
        
        # Convert to structured format
        save_data = {
            "config": {
                "motion_file": self.motion_file,
                "policy_path": self.config.policy_path,
                "control_dt": self.config.control_dt,
                "action_scale": self.config.action_scale,
                "ang_vel_scale": self.config.ang_vel_scale,
                "dof_pos_scale": self.config.dof_pos_scale,
                "dof_vel_scale": self.config.dof_vel_scale,
            },
            "logs": self.obs_log_data,
        }
        
        joblib.dump(save_data, self.obs_log_file)
        print(f"Saved {len(self.obs_log_data)} observation logs to: {self.obs_log_file}")
    
    def _get_motion_frame_idx(self, time_offset=0.0):
        """Get the motion frame index for the current time + offset."""
        current_time = (self.counter * self.config.control_dt + time_offset) % self.motion_len
        frame_idx = int(current_time / self.motion_dt)
        return min(frame_idx, self.motion_num_frames - 1)
    
    def _quat_mul(self, q1, q2):
        """Multiply two quaternions (w, x, y, z format)."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.stack([w, x, y, z], axis=-1)
    
    def _quat_conjugate(self, q):
        """Conjugate of quaternion (w, x, y, z format)."""
        return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)
    
    def _quat_to_rotation_6d(self, quat):
        """Convert quaternion to 6D rotation representation (first two columns of rotation matrix)."""
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # First column of rotation matrix
        r00 = 1 - 2 * (y**2 + z**2)
        r10 = 2 * (x * y + w * z)
        r20 = 2 * (x * z - w * y)
        
        # Second column of rotation matrix
        r01 = 2 * (x * y - w * z)
        r11 = 1 - 2 * (x**2 + z**2)
        r21 = 2 * (y * z + w * x)
        
        return np.stack([r00, r10, r20, r01, r11, r21], axis=-1)
    
    def _quat_to_roll_pitch(self, quat):
        """Convert quaternion (w, x, y, z) to roll and pitch angles."""
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        return np.stack([roll, pitch], axis=-1)
    
    def _get_future_motion_targets(self):
        """Compute future motion targets for student model using motion library."""
        # Sample future timesteps: linspace from 1 to future_max_steps with future_num_steps points
        tar_steps = np.linspace(1, self.future_max_steps, self.future_num_steps, dtype=np.int32)
        
        future_root_height = []
        future_roll_pitch = []
        future_base_lin_vel = []
        future_base_yaw_vel = []
        future_dof_pos = []
        
        for step in tar_steps:
            time_offset = step * self.config.control_dt
            motion_time = torch.tensor(self.counter * self.config.control_dt + time_offset, dtype=torch.float32)
            motion_ids = torch.zeros((1,), dtype=torch.int32)
            motion_res = self.motion_lib.get_motion_state(motion_ids, motion_time)
            
            # Extract data from motion library
            root_pos = motion_res["root_pos"][0].numpy()
            root_rot_xyzw = motion_res["root_rot"][0].numpy()  # XYZW format
            root_vel_world = motion_res["root_vel"][0].numpy()
            root_ang_vel_world = motion_res["root_ang_vel"][0].numpy()
            dof_pos = motion_res["dof_pos"][0].numpy()
            
            # Root height
            future_root_height.append(root_pos[2:3])
            
            # Roll pitch from quaternion (convert XYZW to WXYZ)
            root_rot_wxyz = root_rot_xyzw[[3, 0, 1, 2]]
            roll_pitch = self._quat_to_roll_pitch(root_rot_wxyz)
            future_roll_pitch.append(roll_pitch)
            
            # Root linear velocity (in local frame)
            local_vel = quat_rotate_inverse(root_rot_wxyz[np.newaxis], root_vel_world[np.newaxis])[0]
            future_base_lin_vel.append(local_vel)
            
            # Yaw angular velocity (in local frame)
            local_ang_vel = quat_rotate_inverse(root_rot_wxyz[np.newaxis], root_ang_vel_world[np.newaxis])[0]
            future_base_yaw_vel.append(local_ang_vel[2:3])
            
            # DOF positions
            future_dof_pos.append(dof_pos)
        
        # Concatenate all: (future_num_steps, dim) -> flatten to (1, total_dim)
        # IMPORTANT: Observations must be in SORTED (alphabetical) order!
        # sorted order: future_motion_base_lin_vel, future_motion_base_yaw_vel, future_motion_dof_pos, future_motion_roll_pitch, future_motion_root_height
        future_motion_targets = np.concatenate([
            np.array(future_base_lin_vel).flatten(),     # 3 * 20 = 60
            np.array(future_base_yaw_vel).flatten(),     # 1 * 20 = 20
            np.array(future_dof_pos).flatten(),          # 23 * 20 = 460
            np.array(future_roll_pitch).flatten(),       # 2 * 20 = 40
            np.array(future_root_height).flatten(),      # 1 * 20 = 20
        ], axis=0).astype(np.float32)
        
        return future_motion_targets.reshape(1, -1)
    
    def _get_prop_history(self, base_ang_vel, roll_pitch, dof_pos, dof_vel):
        """Get proprioceptive history for student model.
        History is concatenated in sorted key order: actions, base_ang_vel, dof_pos, dof_vel, roll_pitch
        Each history item has history_length timesteps.
        """
        # Update history (append current to left, oldest falls off right)
        self.student_history["base_ang_vel"].appendleft(base_ang_vel.copy())
        self.student_history["roll_pitch"].appendleft(roll_pitch.copy())
        self.student_history["dof_pos"].appendleft(dof_pos.copy())
        self.student_history["dof_vel"].appendleft(dof_vel.copy())
        self.student_history["actions"].appendleft(self.action.copy())
        
        # Concatenate history in sorted key order
        # For each key, concatenate all timesteps: [t0, t1, ..., t9]
        history_parts = []
        for key in ["actions", "base_ang_vel", "dof_pos", "dof_vel", "roll_pitch"]:
            # Concatenate all timesteps for this key
            key_history = np.concatenate([self.student_history[key][i] for i in range(self.history_length)], axis=0)
            history_parts.append(key_history)
        
        prop_history = np.concatenate(history_parts, axis=0).astype(np.float32)
        return prop_history.reshape(1, -1)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 5
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        self.default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        self.default_angles = self.default_pos[0:23]
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = self.default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    

    def run(self):

        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        
        # 1. Gather individual history tensors
        action_hist_tensor = torch.cat([self.history["action"][i] for i in range(self.frame_stack-1)], dim=1)
        omega_hist_tensor = torch.cat([self.history["omega"][i] for i in range(self.frame_stack-1)], dim=1)
        qj_hist_tensor = torch.cat([self.history["qj"][i] for i in range(self.frame_stack-1)], dim=1)
        dqj_hist_tensor = torch.cat([self.history["dqj"][i] for i in range(self.frame_stack-1)], dim=1)
        gravity_orientation_hist_tensor = torch.cat([self.history["gravity_orientation"][i] for i in range(self.frame_stack-1)], dim=1)
        ref_motion_phase_hist_tensor = torch.cat([self.history["ref_motion_phase"][i] for i in range(self.frame_stack-1)], dim=1)
        
        # 2. Concatenate all parts into a single observation tensor
        obs_hist = torch.cat([
            action_hist_tensor,
            omega_hist_tensor,
            qj_hist_tensor,
            dqj_hist_tensor,
            gravity_orientation_hist_tensor,
            ref_motion_phase_hist_tensor
        ], dim=1)

        # 3. Get the current observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        ref_motion_phase = ((self.counter * self.config.control_dt) % self.motion_len) / self.motion_len

        num_actions = self.config.num_actions

        curr_obs = np.zeros(self.config.num_obs, dtype=np.float32)
        curr_obs[: num_actions] = self.action
        curr_obs[num_actions: num_actions + 3] = ang_vel
        curr_obs[num_actions + 3: 2 * num_actions + 3] = qj_obs
        curr_obs[2 * num_actions + 3: 3 * num_actions + 3] = dqj_obs
        curr_obs[3 * num_actions + 3: 3 * num_actions + 6] = gravity_orientation
        curr_obs[6 + 3 * num_actions] = ref_motion_phase

        curr_obs_tensor = torch.from_numpy(curr_obs).unsqueeze(0)
        
        # 4. Get obs buffer, the order is key's alphabetical order
        self.obs_buf = torch.cat([
            curr_obs_tensor[:, :3 * num_actions + 3], 
            obs_hist, 
            curr_obs_tensor[:, 3 * num_actions + 3:]], 
            dim=1
        )

        # 5. Update the history (for teacher model)
        self.history["action"].appendleft(curr_obs_tensor[:, :num_actions])
        self.history["omega"].appendleft(curr_obs_tensor[:, num_actions:num_actions+3])
        self.history["qj"].appendleft(curr_obs_tensor[:, num_actions+3:num_actions+3+num_actions])
        self.history["dqj"].appendleft(curr_obs_tensor[:, num_actions+3+num_actions:num_actions+3+2*num_actions])
        self.history["gravity_orientation"].appendleft(curr_obs_tensor[:, num_actions+3+2*num_actions:num_actions+3+2*num_actions+3])
        self.history["ref_motion_phase"].appendleft(curr_obs_tensor[:, -1].unsqueeze(0))
        
        # 6. Get policy's inference
        if self.is_student_model:
            # Student model: requires actor_obs, future_motion_targets, prop_history
            
            # Compute roll_pitch
            roll_pitch = np.array([
                np.arctan2(2.0 * (quat[0] * quat[1] + quat[2] * quat[3]), 
                          1.0 - 2.0 * (quat[1]**2 + quat[2]**2)),
                np.arcsin(np.clip(2.0 * (quat[0] * quat[2] - quat[3] * quat[1]), -1.0, 1.0))
            ], dtype=np.float32)
            
            # Get future motion targets
            future_motion_targets = self._get_future_motion_targets()
            
            # Get prop_history (also updates history)
            prop_history = self._get_prop_history(
                base_ang_vel=ang_vel.flatten(),
                roll_pitch=roll_pitch,
                dof_pos=qj_obs,
                dof_vel=dqj_obs
            )
            
            # Build actor_obs for student model (877 dims):
            # 1. base_ang_vel: 3
            # 2. dof_pos: 23
            # 3. dof_vel: 23  
            # 4. actions: 23
            # 5. roll_pitch: 2
            # 6. anchor_ref_rot: 6 (6D rotation representation)
            # 7. next_step_ref_motion: 57
            # 8. history: 740 (same as prop_history)
            
            # Get next step reference motion from motion library
            motion_time = torch.tensor((self.counter + 1) * self.config.control_dt, dtype=torch.float32)
            motion_ids = torch.zeros((1,), dtype=torch.int32)
            motion_res = self.motion_lib.get_motion_state(motion_ids, motion_time)
            
            # Extract data from motion library result
            next_root_pos = motion_res["root_pos"][0].numpy()  # (3,)
            next_root_rot_xyzw = motion_res["root_rot"][0].numpy()  # (4,) XYZW format
            next_root_vel_world = motion_res["root_vel"][0].numpy()  # (3,)
            next_root_ang_vel_world = motion_res["root_ang_vel"][0].numpy()  # (3,)
            next_dof_pos = motion_res["dof_pos"][0].numpy()  # (23,)
            ref_body_pos = motion_res["rg_pos_t"][0].numpy()  # (num_bodies, 3)
            ref_body_rot = motion_res["rg_rot_t"][0].numpy()  # (num_bodies, 4) XYZW
            
            # Root height
            next_root_height = next_root_pos[2:3]
            
            # Convert quaternion XYZW to WXYZ for roll_pitch calculation
            next_root_rot_wxyz = next_root_rot_xyzw[[3, 0, 1, 2]]
            next_roll_pitch = self._quat_to_roll_pitch(next_root_rot_wxyz)
            
            # Root velocity in local frame (use WXYZ for quat_rotate_inverse)
            next_local_vel = quat_rotate_inverse(next_root_rot_wxyz[np.newaxis], next_root_vel_world[np.newaxis])[0]
            
            # Root angular velocity yaw (local frame)
            next_local_ang_vel = quat_rotate_inverse(next_root_rot_wxyz[np.newaxis], next_root_ang_vel_world[np.newaxis])[0]
            next_yaw_vel = next_local_ang_vel[2:3]
            
            # Compute local key body positions relative to anchor (root)
            anchor_pos = ref_body_pos[self.anchor_index]  # (3,)
            anchor_rot_xyzw = ref_body_rot[self.anchor_index]  # (4,) XYZW
            anchor_rot_wxyz = anchor_rot_xyzw[[3, 0, 1, 2]]
            
            # Get key body positions and transform to local frame
            key_body_pos_world = ref_body_pos[self.key_body_id]  # (9, 3)
            key_body_pos_relative = key_body_pos_world - anchor_pos  # (9, 3)
            
            # Rotate to local frame using anchor inverse rotation
            next_key_body_pos = quat_rotate_inverse(
                np.tile(anchor_rot_wxyz, (len(self.key_body_id), 1)),
                key_body_pos_relative
            ).flatten().astype(np.float32)  # (27,)
            
            next_step_ref_motion = np.concatenate([
                next_root_height.astype(np.float32),      # 1
                next_roll_pitch.astype(np.float32),       # 2
                next_local_vel.astype(np.float32),        # 3
                next_yaw_vel.astype(np.float32),          # 1
                next_dof_pos.astype(np.float32),          # 23
                next_key_body_pos,                        # 27
            ], axis=0)
            
            # Compute anchor_ref_rot: 6D rotation representation (first 2 columns of rotation matrix)
            # This represents the relative rotation from current robot to reference motion
            # Use reference motion orientation relative to current robot orientation
            current_quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)  # Convert WXYZ to XYZW
            ref_quat_xyzw = next_root_rot_xyzw  # Already in XYZW format from motion library
            
            # Compute relative rotation: current_inv * ref
            # quat_inv for XYZW: [-x, -y, -z, w]
            current_quat_inv = np.array([-current_quat_xyzw[0], -current_quat_xyzw[1], -current_quat_xyzw[2], current_quat_xyzw[3]])
            
            # Simple quaternion multiplication (not perfectly accurate but close enough)
            # For now, use a simpler approximation: convert to rotation matrix
            def quat_to_rotmat(q):
                """Convert XYZW quaternion to 3x3 rotation matrix."""
                x, y, z, w = q[0], q[1], q[2], q[3]
                return np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                ], dtype=np.float32)
            
            current_rotmat = quat_to_rotmat(current_quat_xyzw)
            ref_rotmat = quat_to_rotmat(ref_quat_xyzw)
            
            # Relative rotation: current.T @ ref
            rel_rotmat = current_rotmat.T @ ref_rotmat
            
            # Take first 2 columns (6D representation)
            anchor_ref_rot_6d = rel_rotmat[:, :2].flatten()  # 6D
            
            # IMPORTANT: Observations must be in SORTED (alphabetical) order!
            # sorted order: actions, anchor_ref_rot, base_ang_vel, dof_pos, dof_vel, history, next_step_ref_motion, roll_pitch
            actor_obs = np.concatenate([
                self.action,           # actions: 23
                anchor_ref_rot_6d,     # anchor_ref_rot: 6
                ang_vel.flatten(),     # base_ang_vel: 3
                qj_obs,                # dof_pos: 23
                dqj_obs,               # dof_vel: 23
                prop_history.flatten(),# history: 740
                next_step_ref_motion,  # next_step_ref_motion: 57
                roll_pitch,            # roll_pitch: 2
            ], axis=0).astype(np.float32).reshape(1, -1)
            
            # Prepare inputs
            inputs = {
                "actor_obs": actor_obs,
                "future_motion_targets": future_motion_targets,
                "prop_history": prop_history
            }
            
            # Debug: print dimensions on first run
            if self.counter == 1:
                print(f"Debug - Input dimensions:")
                print(f"  actor_obs: {actor_obs.shape} (expected: [1, 877])")
                print(f"  future_motion_targets: {future_motion_targets.shape} (expected: [1, 600])")
                print(f"  prop_history: {prop_history.shape} (expected: [1, 740])")
            
            # Log observations if enabled
            if self.obs_logging_enabled and (self.counter % self.obs_log_interval == 0):
                self._log_observation(
                    actor_obs=actor_obs,
                    future_motion_targets=future_motion_targets,
                    prop_history=prop_history,
                    raw_obs={
                        "actions": self.action.copy(),
                        "anchor_ref_rot": anchor_ref_rot_6d.copy(),
                        "base_ang_vel": ang_vel.flatten().copy(),
                        "dof_pos": qj_obs.copy(),
                        "dof_vel": dqj_obs.copy(),
                        "roll_pitch": roll_pitch.copy(),
                        "next_step_ref_motion": next_step_ref_motion.copy(),
                        "quat": np.array(quat),
                        "gravity_orientation": gravity_orientation.copy(),
                    }
                )
            
            outputs = self.policy.run(None, inputs)
        else:
            # Teacher model: only requires actor_obs
            input_name = self.policy.get_inputs()[0].name
            outputs = self.policy.run(None, {input_name: self.obs_buf.numpy()})
        
        self.action = outputs[0].squeeze()
        target_dof_pos = self.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    parser.add_argument("--log-obs", action="store_true", help="Enable observation logging")
    parser.add_argument("--log-interval", type=int, default=10, help="Log observation every N steps")
    args = parser.parse_args()

    # Load config
    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)
    
    # Override config with command line args
    config.obs_logging = args.log_obs
    config.obs_log_interval = args.log_interval

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    
    # Save observation log if enabled
    controller.save_obs_log()
    
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
