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


def calc_heading_quat_xyzw(q_xyzw):
    """Calculate heading quaternion from quaternion (XYZW format).
    Returns only the yaw rotation component."""
    x, y, z, w = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]
    heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    half_heading = heading * 0.5
    heading_quat = np.stack([
        np.zeros_like(half_heading),  # x
        np.zeros_like(half_heading),  # y
        np.sin(half_heading),         # z
        np.cos(half_heading)          # w
    ], axis=-1)
    return heading_quat.astype(np.float32)


def quat_inverse_xyzw(q_xyzw):
    """Inverse of quaternion in XYZW format."""
    return np.array([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]], dtype=np.float32)


def quat_mul_xyzw(q1, q2):
    """Multiply two quaternions in XYZW format."""
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float32)


def quat_apply_xyzw(q, v):
    """Apply quaternion rotation to vector. Quaternion in XYZW format."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def remove_yaw_offset(quat_xyzw, heading_quat_xyzw):
    """Remove yaw offset from quaternion.
    quat * heading_inv = quaternion with yaw removed.
    """
    heading_inv = quat_inverse_xyzw(heading_quat_xyzw)
    return quat_mul_xyzw(quat_xyzw, heading_inv)


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.frame_stack = self.config.frame_stack
        
        # Debug logging settings
        self.debug_log_enabled = getattr(config, 'debug_log', False)
        self.debug_log_interval = getattr(config, 'debug_log_interval', 50)
        
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
        
        # Init frame variables for yaw offset alignment (student model - Method 2)
        self.init_frame_set = False
        self.robot_yaw_offset = None  # Robot's initial heading quaternion (XYZW)
        self.motion_yaw_offset = None  # Motion's initial heading quaternion (XYZW)
        
        # Initial dof_pos offset compensation (to handle imperfect default pose)
        self.init_dof_pos_offset = None
        
        # Initial roll/pitch offset compensation (robot cannot achieve perfect upright pose)
        self.init_roll_pitch_offset = None

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
        
        # Observation logging setup (pkl file)
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
        
        # Debug text logging setup (txt file)
        self.debug_log_file = None
        if self.debug_log_enabled:
            log_dir = os.path.join("logs", "debug_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.debug_log_filename = os.path.join(log_dir, f"real_world_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            self.debug_log_file = open(self.debug_log_filename, 'w')
            print(f"Debug logging enabled. Will save to: {self.debug_log_filename}")
    
    def _debug_log(self, msg):
        """Write debug message to log file."""
        if self.debug_log_file:
            self.debug_log_file.write(msg + '\n')
            self.debug_log_file.flush()  # Ensure immediate write

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
    
    def _setup_init_frame(self, robot_quat_xyzw):
        """Setup initial frames for reference-to-robot frame transformation.
        Called once on the first frame to capture robot and motion initial orientations.
        This follows the same approach as sim-to-sim (urcirobot.py).
        """
        # Store robot initial orientation (XYZW)
        self.robot_init_rot_xyzw = robot_quat_xyzw.copy()
        
        # Motion initial orientation - get from t=0 (first frame of motion)
        motion_ids = torch.zeros((1,), dtype=torch.int32)
        motion_time_zero = torch.tensor(0.0, dtype=torch.float32)
        motion_res_init = self.motion_lib.get_motion_state(motion_ids, motion_time_zero)
        self.ref_init_rot_xyzw = motion_res_init["root_rot"][0].numpy()  # (4,) XYZW
        self.ref_init_pos = motion_res_init["root_pos"][0].numpy()  # (3,)
        ref_init_dof_pos = motion_res_init["dof_pos"][0].numpy()
        
        # Compute q_rel: relative rotation from motion frame to robot frame
        # q_rel = robot_init_rot * ref_init_rot.inverse()
        # This transforms any motion quaternion to robot frame: q_new = q_rel * ref_quat
        ref_init_inv = quat_inverse_xyzw(self.ref_init_rot_xyzw)
        self.q_rel = quat_mul_xyzw(self.robot_init_rot_xyzw, ref_init_inv)
        
        self.init_frame_set = True
        
        # Compute yaw difference for debugging
        robot_yaw = 2 * np.arctan2(robot_quat_xyzw[2], robot_quat_xyzw[3])
        motion_yaw = 2 * np.arctan2(self.ref_init_rot_xyzw[2], self.ref_init_rot_xyzw[3])
        yaw_diff_deg = np.degrees(robot_yaw - motion_yaw)
        
        # Compute robot's initial roll/pitch from quaternion (XYZW)
        x, y, z, w = robot_quat_xyzw[0], robot_quat_xyzw[1], robot_quat_xyzw[2], robot_quat_xyzw[3]
        robot_roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        robot_pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        
        # Compute motion's initial roll/pitch
        mx, my, mz, mw = self.ref_init_rot_xyzw[0], self.ref_init_rot_xyzw[1], self.ref_init_rot_xyzw[2], self.ref_init_rot_xyzw[3]
        motion_roll = np.arctan2(2.0 * (mw * mx + my * mz), 1.0 - 2.0 * (mx**2 + my**2))
        motion_pitch = np.arcsin(np.clip(2.0 * (mw * my - mz * mx), -1.0, 1.0))
        
        self._debug_log(f"{'='*60}")
        self._debug_log(f"[INIT_FRAME] INITIALIZATION (ref-to-robot transform)")
        self._debug_log(f"{'='*60}")
        self._debug_log(f"[INIT_FRAME] robot_init_quat(XYZW)=[{robot_quat_xyzw[0]:.4f}, {robot_quat_xyzw[1]:.4f}, {robot_quat_xyzw[2]:.4f}, {robot_quat_xyzw[3]:.4f}]")
        self._debug_log(f"[INIT_FRAME] motion_init_quat(XYZW)=[{self.ref_init_rot_xyzw[0]:.4f}, {self.ref_init_rot_xyzw[1]:.4f}, {self.ref_init_rot_xyzw[2]:.4f}, {self.ref_init_rot_xyzw[3]:.4f}]")
        self._debug_log(f"[INIT_FRAME] q_rel(XYZW)=[{self.q_rel[0]:.4f}, {self.q_rel[1]:.4f}, {self.q_rel[2]:.4f}, {self.q_rel[3]:.4f}]")
        self._debug_log(f"[INIT_FRAME] robot_yaw={np.degrees(robot_yaw):.1f}°, motion_yaw={np.degrees(motion_yaw):.1f}°")
        self._debug_log(f"[INIT_FRAME] yaw_diff={yaw_diff_deg:.1f}° (will be handled by q_rel transform)")
        self._debug_log(f"[INIT_FRAME] robot_roll_pitch=[{np.degrees(robot_roll):.2f}°, {np.degrees(robot_pitch):.2f}°]")
        self._debug_log(f"[INIT_FRAME] motion_roll_pitch=[{np.degrees(motion_roll):.2f}°, {np.degrees(motion_pitch):.2f}°]")
        self._debug_log(f"[INIT_FRAME] roll_pitch_diff=[{np.degrees(robot_roll - motion_roll):.2f}°, {np.degrees(robot_pitch - motion_pitch):.2f}°]")
        self._debug_log(f"[INIT_FRAME] motion_init_dof_pos range=[{ref_init_dof_pos.min():.4f}, {ref_init_dof_pos.max():.4f}]")
        self._debug_log(f"[INIT_FRAME] default_angles range=[{self.default_angles.min():.4f}, {self.default_angles.max():.4f}]")
        dof_diff = self.default_angles - ref_init_dof_pos
        self._debug_log(f"[INIT_FRAME] default - motion_init_dof: range=[{dof_diff.min():.4f}, {dof_diff.max():.4f}], mean={dof_diff.mean():.4f}")
        self._debug_log(f"{'='*60}\n")
        print(f"[INIT_FRAME] Yaw diff: {yaw_diff_deg:.1f}° (handled by q_rel transform)")
        print(f"[INIT_FRAME] Robot roll/pitch: [{np.degrees(robot_roll):.2f}°, {np.degrees(robot_pitch):.2f}°]")
    
    def _ref_to_robot_frame(self, ref_quat_xyzw):
        """Transform reference motion quaternion to robot frame.
        This is equivalent to fn_ref_to_robot_frame in urcirobot.py.
        q_new = q_rel * ref_quat
        """
        return quat_mul_xyzw(self.q_rel, ref_quat_xyzw)
    
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
                "action_scale": self.config.action_scale.tolist() if isinstance(self.config.action_scale, np.ndarray) else self.config.action_scale,
                "ang_vel_scale": self.config.ang_vel_scale,
                "dof_pos_scale": self.config.dof_pos_scale,
                "dof_vel_scale": self.config.dof_vel_scale,
            },
            "logs": self.obs_log_data,
        }
        
        joblib.dump(save_data, self.obs_log_file)
        print(f"Saved {len(self.obs_log_data)} observation logs to: {self.obs_log_file}")
    
    def close_debug_log(self):
        """Close debug log file."""
        if self.debug_log_file:
            self.debug_log_file.close()
            print(f"Debug log saved to: {self.debug_log_filename}")
    
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
            # Use (counter - 1) to align with sim-to-sim where timer starts at 0
            # counter=1 -> base_time=0, counter=2 -> base_time=0.02, etc.
            motion_time = torch.tensor((self.counter - 1) * self.config.control_dt + time_offset, dtype=torch.float32)
            motion_ids = torch.zeros((1,), dtype=torch.int32)
            motion_res = self.motion_lib.get_motion_state(motion_ids, motion_time)
            
            # Extract data from motion library (use original motion frame)
            root_pos = motion_res["root_pos"][0].numpy()
            root_rot_xyzw = motion_res["root_rot"][0].numpy()  # XYZW format
            root_vel_world = motion_res["root_vel"][0].numpy()
            root_ang_vel_world = motion_res["root_ang_vel"][0].numpy()
            dof_pos = motion_res["dof_pos"][0].numpy()
            
            # Method 2: Use motion's original orientation without transformation
            # Local velocities are computed in motion's local frame, which represents
            # "forward/sideways/up" relative to motion's body orientation.
            # Robot will interpret these as its own local directions.
            
            # Root height (unchanged - vertical position doesn't depend on yaw)
            future_root_height.append(root_pos[2:3])
            
            # Roll pitch from motion quaternion (convert XYZW to WXYZ)
            # Note: roll/pitch are relative to motion's body, not world frame
            root_rot_wxyz = root_rot_xyzw[[3, 0, 1, 2]]
            roll_pitch = self._quat_to_roll_pitch(root_rot_wxyz)
            future_roll_pitch.append(roll_pitch)
            
            # Root linear velocity (in motion's local frame)
            local_vel = quat_rotate_inverse(root_rot_wxyz[np.newaxis], root_vel_world[np.newaxis])[0]
            future_base_lin_vel.append(local_vel)
            
            # Yaw angular velocity (in motion's local frame)
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
        
        # Normalize quaternion: ensure w is positive (canonical form)
        # This is important because q and -q represent the same rotation
        if quat[0] < 0:
            quat = tuple(-x for x in quat)
        
        # Capture initial offsets on first step (BEFORE computing observations)
        if self.init_dof_pos_offset is None and self.counter == 1:
            self.init_dof_pos_offset = (self.qj - self.default_angles).copy()
            print(f"Initial dof_pos offset captured: range=[{self.init_dof_pos_offset.min():.4f}, {self.init_dof_pos_offset.max():.4f}]")
            
            # Also capture roll/pitch offset (robot cannot be perfectly upright)
            temp_roll_pitch = self._quat_to_roll_pitch(np.array(quat))
            self.init_roll_pitch_offset = temp_roll_pitch.copy()
            print(f"Initial roll/pitch offset captured: [{np.degrees(temp_roll_pitch[0]):.2f}°, {np.degrees(temp_roll_pitch[1]):.2f}°]")

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
        
        # Compensate for initial pose offset (makes dof_pos observation relative to actual initial pose)
        if self.init_dof_pos_offset is not None:
            qj_obs = (qj_obs - self.default_angles - self.init_dof_pos_offset) * self.config.dof_pos_scale
        else:
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
            
            # Compensate for initial roll/pitch offset (robot cannot be perfectly upright)
            if self.init_roll_pitch_offset is not None:
                roll_pitch = roll_pitch - self.init_roll_pitch_offset
            
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
            
            # Setup init frame on first run (for ref-to-robot transformation)
            if not self.init_frame_set:
                # Convert current robot quat from WXYZ to XYZW
                robot_quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
                self._setup_init_frame(robot_quat_xyzw)
            
            # Extract data from motion library result (use original motion frame)
            # Method 2: No transformation - use motion's original orientation
            next_root_pos = motion_res["root_pos"][0].numpy()  # (3,)
            next_root_rot_xyzw = motion_res["root_rot"][0].numpy()  # (4,) XYZW format
            next_root_vel_world = motion_res["root_vel"][0].numpy()  # (3,)
            next_root_ang_vel_world = motion_res["root_ang_vel"][0].numpy()  # (3,)
            next_dof_pos = motion_res["dof_pos"][0].numpy()  # (23,)
            ref_body_pos = motion_res["rg_pos_t"][0].numpy()  # (num_bodies, 3)
            ref_body_rot = motion_res["rg_rot_t"][0].numpy()  # (num_bodies, 4) XYZW
            
            # Root height (unchanged - vertical position doesn't depend on yaw)
            next_root_height = next_root_pos[2:3]
            
            # Convert motion quaternion XYZW to WXYZ for calculations
            next_root_rot_wxyz = next_root_rot_xyzw[[3, 0, 1, 2]]
            next_roll_pitch = self._quat_to_roll_pitch(next_root_rot_wxyz)
            
            # Root velocity in motion's local frame
            next_local_vel = quat_rotate_inverse(next_root_rot_wxyz[np.newaxis], next_root_vel_world[np.newaxis])[0]
            
            # Root angular velocity yaw (in motion's local frame)
            next_local_ang_vel = quat_rotate_inverse(next_root_rot_wxyz[np.newaxis], next_root_ang_vel_world[np.newaxis])[0]
            next_yaw_vel = next_local_ang_vel[2:3]
            
            # Compute local key body positions relative to anchor (root)
            # Use motion's original rotation (relative positions are computed in motion's local frame)
            anchor_pos = ref_body_pos[self.anchor_index]  # (3,)
            anchor_rot_xyzw = ref_body_rot[self.anchor_index]  # (4,) XYZW
            anchor_rot_wxyz = anchor_rot_xyzw[[3, 0, 1, 2]]
            
            # Get key body positions (relative positions are invariant to yaw rotation)
            key_body_pos_world = ref_body_pos[self.key_body_id]  # (9, 3)
            key_body_pos_relative = key_body_pos_world - anchor_pos  # (9, 3)
            
            # Rotate to local frame using motion's anchor orientation
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
            # This follows the same approach as sim-to-sim (urcirobot.py):
            # anchor_ref_rot = matrix_from_quat(quat_inverse(robot_quat) * ref_to_robot_frame(ref_quat))
            current_quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)  # Convert WXYZ to XYZW
            
            # Transform ref motion quaternion to robot frame
            ref_quat_in_robot_frame = self._ref_to_robot_frame(next_root_rot_xyzw)
            
            # Compute relative rotation: robot_quat.inverse() * ref_quat_in_robot_frame
            robot_quat_inv = quat_inverse_xyzw(current_quat_xyzw)
            rel_quat = quat_mul_xyzw(robot_quat_inv, ref_quat_in_robot_frame)
            
            def quat_to_rotmat(q):
                """Convert XYZW quaternion to 3x3 rotation matrix."""
                x, y, z, w = q[0], q[1], q[2], q[3]
                return np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                ], dtype=np.float32)
            
            # Convert relative quaternion to rotation matrix
            rel_rotmat = quat_to_rotmat(rel_quat)
            
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
                print(f"[DEBUG] Input dimensions:")
                print(f"  actor_obs: {actor_obs.shape} (expected: [1, 877])")
                print(f"  future_motion_targets: {future_motion_targets.shape} (expected: [1, 600])")
                print(f"  prop_history: {prop_history.shape} (expected: [1, 740])")
            
            # Detailed debug logging to file
            if self.debug_log_enabled and (self.counter % self.debug_log_interval == 0 or self.counter <= 3):
                self._debug_log(f"\n{'='*60}")
                self._debug_log(f"[STEP {self.counter}] t={self.counter * self.config.control_dt:.3f}s, motion_time={motion_time.item():.3f}s")
                self._debug_log(f"{'='*60}")
                
                # IMU raw data
                self._debug_log(f"[IMU] quat_raw(WXYZ)=[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
                self._debug_log(f"[IMU] ang_vel=[{ang_vel.flatten()[0]:.4f}, {ang_vel.flatten()[1]:.4f}, {ang_vel.flatten()[2]:.4f}]")
                self._debug_log(f"[IMU] roll_pitch (offset-compensated)=[{roll_pitch[0]:.4f}, {roll_pitch[1]:.4f}] (deg: [{np.degrees(roll_pitch[0]):.2f}, {np.degrees(roll_pitch[1]):.2f}])")
                if self.init_roll_pitch_offset is not None:
                    self._debug_log(f"[IMU] init_roll_pitch_offset=[{self.init_roll_pitch_offset[0]:.4f}, {self.init_roll_pitch_offset[1]:.4f}] (deg: [{np.degrees(self.init_roll_pitch_offset[0]):.2f}, {np.degrees(self.init_roll_pitch_offset[1]):.2f}])")
                
                # Ref-to-robot frame transformation - IMPORTANT for debugging
                robot_yaw_current = 2 * np.arctan2(current_quat_xyzw[2], current_quat_xyzw[3])
                motion_yaw_raw = 2 * np.arctan2(next_root_rot_xyzw[2], next_root_rot_xyzw[3])
                ref_in_robot_yaw = 2 * np.arctan2(ref_quat_in_robot_frame[2], ref_quat_in_robot_frame[3])
                self._debug_log(f"[YAW] robot_current={np.degrees(robot_yaw_current):.1f}°, motion_raw={np.degrees(motion_yaw_raw):.1f}°")
                self._debug_log(f"[YAW] ref_in_robot_frame={np.degrees(ref_in_robot_yaw):.1f}°")
                self._debug_log(f"[YAW] yaw_diff(robot - ref_in_robot)={np.degrees(robot_yaw_current - ref_in_robot_yaw):.1f}°")
                
                # Quaternions for debugging
                self._debug_log(f"[QUAT] robot_quat(XYZW)=[{current_quat_xyzw[0]:.4f}, {current_quat_xyzw[1]:.4f}, {current_quat_xyzw[2]:.4f}, {current_quat_xyzw[3]:.4f}]")
                self._debug_log(f"[QUAT] ref_in_robot_frame(XYZW)=[{ref_quat_in_robot_frame[0]:.4f}, {ref_quat_in_robot_frame[1]:.4f}, {ref_quat_in_robot_frame[2]:.4f}, {ref_quat_in_robot_frame[3]:.4f}]")
                self._debug_log(f"[QUAT] rel_quat(XYZW)=[{rel_quat[0]:.4f}, {rel_quat[1]:.4f}, {rel_quat[2]:.4f}, {rel_quat[3]:.4f}]")
                
                # Joint positions - raw and scaled
                qj_raw = self.qj - self.default_angles  # actual raw joint positions (relative to default)
                self._debug_log(f"[JOINT] qj_raw (rad) range=[{qj_raw.min():.4f}, {qj_raw.max():.4f}]")
                self._debug_log(f"[JOINT] qj_obs (scaled, offset-compensated) range=[{qj_obs.min():.4f}, {qj_obs.max():.4f}]")
                self._debug_log(f"[JOINT] dqj_obs (scaled) range=[{dqj_obs.min():.4f}, {dqj_obs.max():.4f}]")
                if self.init_dof_pos_offset is not None:
                    self._debug_log(f"[JOINT] init_offset range=[{self.init_dof_pos_offset.min():.4f}, {self.init_dof_pos_offset.max():.4f}]")
                
                # Reference motion data
                self._debug_log(f"[REF_MOTION] next_root_height={next_root_height[0]:.4f}")
                self._debug_log(f"[REF_MOTION] next_roll_pitch=[{next_roll_pitch[0]:.4f}, {next_roll_pitch[1]:.4f}] (deg: [{np.degrees(next_roll_pitch[0]):.2f}, {np.degrees(next_roll_pitch[1]):.2f}])")
                self._debug_log(f"[REF_MOTION] next_local_vel=[{next_local_vel[0]:.4f}, {next_local_vel[1]:.4f}, {next_local_vel[2]:.4f}]")
                self._debug_log(f"[REF_MOTION] next_key_body_pos range=[{next_key_body_pos.min():.4f}, {next_key_body_pos.max():.4f}]")
                
                # Compare robot vs ref dof_pos
                dof_pos_diff = qj_raw - next_dof_pos
                self._debug_log(f"[REF_MOTION] next_dof_pos range=[{next_dof_pos.min():.4f}, {next_dof_pos.max():.4f}]")
                self._debug_log(f"[DOF_DIFF] robot_dof - ref_dof: range=[{dof_pos_diff.min():.4f}, {dof_pos_diff.max():.4f}], mean={dof_pos_diff.mean():.4f}")
                
                # Roll/pitch comparison
                roll_diff = roll_pitch[0] - next_roll_pitch[0]
                pitch_diff = roll_pitch[1] - next_roll_pitch[1]
                self._debug_log(f"[POSE_DIFF] roll_diff={np.degrees(roll_diff):.2f}°, pitch_diff={np.degrees(pitch_diff):.2f}°")
                
                # Anchor rotation
                self._debug_log(f"[ANCHOR_ROT] anchor_ref_rot_6d=[{anchor_ref_rot_6d[0]:.4f}, {anchor_ref_rot_6d[1]:.4f}, {anchor_ref_rot_6d[2]:.4f}, {anchor_ref_rot_6d[3]:.4f}, {anchor_ref_rot_6d[4]:.4f}, {anchor_ref_rot_6d[5]:.4f}]")
                
                # Input ranges
                self._debug_log(f"[FUTURE] future_motion_targets range=[{future_motion_targets.min():.4f}, {future_motion_targets.max():.4f}]")
                self._debug_log(f"[HISTORY] prop_history range=[{prop_history.min():.4f}, {prop_history.max():.4f}]")
            
            outputs = self.policy.run(None, inputs)
        else:
            # Teacher model: only requires actor_obs
            input_name = self.policy.get_inputs()[0].name
            outputs = self.policy.run(None, {input_name: self.obs_buf.numpy()})
        
        self.action = outputs[0].squeeze()
        
        # Clip action to prevent instability from policy sensitivity to small IMU/orientation differences
        # Student model trained in sim shows actions typically in [-0.3, 0.3] range
        # Real-world IMU differences (~1-2° in roll/pitch) can cause 10x larger actions, leading to cascading instability
        action_clip_limit = 0.6  # Conservative limit (sim max is ~0.2-0.3 typically)
        if np.abs(self.action).max() > action_clip_limit:
            print(f"[WARNING] Action clipped from [{self.action.min():.3f}, {self.action.max():.3f}] to [{-action_clip_limit:.3f}, {action_clip_limit:.3f}]")
        self.action = np.clip(self.action, -action_clip_limit, action_clip_limit)
        
        # Log observations AFTER policy execution (so action matches the obs)
        if self.is_student_model and self.obs_logging_enabled and (self.counter % self.obs_log_interval == 0):
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
        target_dof_pos = self.default_angles + self.action * self.config.action_scale
        
        # Debug logging for action output
        if self.is_student_model and self.debug_log_enabled and (self.counter % self.debug_log_interval == 0 or self.counter <= 3):
            self._debug_log(f"[ACTION] raw action range=[{self.action.min():.4f}, {self.action.max():.4f}]")
            
            # Log action_scale info
            if isinstance(self.config.action_scale, np.ndarray):
                self._debug_log(f"[ACTION] action_scale (per-joint) range=[{self.config.action_scale.min():.4f}, {self.config.action_scale.max():.4f}]")
            else:
                self._debug_log(f"[ACTION] action_scale={self.config.action_scale}")
            
            self._debug_log(f"[ACTION] target_dof_pos range=[{target_dof_pos.min():.4f}, {target_dof_pos.max():.4f}]")
            self._debug_log(f"[ACTION] default_angles range=[{self.default_angles.min():.4f}, {self.default_angles.max():.4f}]")
            
            # Detailed per-joint action output
            joint_names = getattr(self.config, 'joint_names', None)
            if joint_names is None:
                joint_names = [f"joint_{i}" for i in range(len(self.action))]
            
            self._debug_log(f"[ACTION_DETAIL] Per-joint actions (raw, scaled, target, default):")
            # Left leg (0-5), Right leg (6-11), Waist (12-14), Left arm (15-18), Right arm (19-22)
            leg_groups = [
                ("L_Leg", slice(0, 6)),
                ("R_Leg", slice(6, 12)),
                ("Waist", slice(12, 15)),
                ("L_Arm", slice(15, 19)),
                ("R_Arm", slice(19, 23)),
            ]
            for group_name, idx_slice in leg_groups:
                raw_act = self.action[idx_slice]
                # Handle both scalar and per-joint action_scale
                if isinstance(self.config.action_scale, np.ndarray):
                    scaled_act = raw_act * self.config.action_scale[idx_slice]
                    scale_info = self.config.action_scale[idx_slice]
                else:
                    scaled_act = raw_act * self.config.action_scale
                    scale_info = self.config.action_scale
                target = target_dof_pos[idx_slice]
                default = self.default_angles[idx_slice]
                self._debug_log(f"  {group_name}: raw={np.array2string(raw_act, precision=2, separator=',')}")
                self._debug_log(f"         target={np.array2string(target, precision=3, separator=',')}, default={np.array2string(default, precision=2, separator=',')}")
            
            # Find joints with largest actions
            max_action_idx = np.argmax(np.abs(self.action))
            self._debug_log(f"[ACTION_MAX] Largest action at joint {max_action_idx}: raw={self.action[max_action_idx]:.4f}, target={target_dof_pos[max_action_idx]:.4f}")
            
            self._debug_log(f"{'='*60}\n")

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
    parser.add_argument("--log-obs", action="store_true", help="Enable observation logging to pkl file")
    parser.add_argument("--log-interval", type=int, default=10, help="Log observation every N steps (for --log-obs)")
    parser.add_argument("--log", action="store_true", help="Enable debug logging to txt file")
    parser.add_argument("--log-debug-interval", type=int, default=1, help="Log debug info every N steps (for --log)")
    args = parser.parse_args()

    # Load config
    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)
    
    # Override config with command line args
    config.obs_logging = args.log_obs
    config.obs_log_interval = args.log_interval
    config.debug_log = args.log
    config.debug_log_interval = args.log_debug_interval

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
    
    # Close debug log file
    controller.close_debug_log()
    
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
