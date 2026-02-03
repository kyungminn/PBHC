"""
Evaluation script for collecting trajectory metrics:
1. Energy (per joint and average)
2. Foot slippage (left, right, average)
3. Foot contact force (left, right, average)
4. Action rate (per joint and average)
5. Motion tracking (per joint global translation distance - min, max, average)

Usage: 
    python humanoidverse/eval_metrics.py \
        +checkpoint=/path/to/model.pt \
        +device=cuda:0 \
        +headless=True
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from utils.config_utils import *  # noqa: E402, F403
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger
import isaacgym
import torch
import numpy as np


class MetricsCollector:
    """Collects and stores trajectory metrics during evaluation."""
    
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.num_dofs = env.num_dofs
        self.dof_names = env.dof_names
        self.feet_indices = env.feet_indices
        
        # Get body names for motion tracking
        self.body_names = env.simulator._body_list if hasattr(env.simulator, '_body_list') else []
        
        # Initialize storage for metrics per step
        self.reset_buffers()
        
    def reset_buffers(self):
        """Reset all metric buffers."""
        # Energy: |torque * joint_vel| per joint per step
        self.energy_per_step = []  # List of [num_envs, num_dofs] tensors
        
        # Foot slippage: xy velocity when in contact
        self.foot_slippage_left_per_step = []  # List of [num_envs] tensors
        self.foot_slippage_right_per_step = []  # List of [num_envs] tensors
        
        # Foot contact force: force magnitude for each foot
        self.foot_contact_force_left_per_step = []  # List of [num_envs] tensors
        self.foot_contact_force_right_per_step = []  # List of [num_envs] tensors
        
        # Action rate: |action_t - action_t-1| per joint per step
        self.action_rate_per_step = []  # List of [num_envs, num_dofs] tensors
        self.prev_actions = None  # Track previous actions manually
        self.prev_prev_actions = None  # Track t-2 actions for smoothness
        
        # Action smoothness (jerk): (action_t - 2*action_t-1 + action_t-2)^2
        self.action_smoothness_per_step = []  # List of [num_envs, num_dofs] tensors
        
        # Joint acceleration: |dof_vel_t - dof_vel_t-1| / dt
        self.joint_acc_per_step = []  # List of [num_envs, num_dofs] tensors
        self.prev_dof_vel = None
        
        # Base angular velocity tracking error
        self.base_ang_vel_error_per_step = []  # List of [num_envs] tensors
        
        # Reference energy (PD approximation)
        self.ref_energy_per_step = []  # List of [num_envs, num_dofs] tensors
        
        # Motion tracking (global frame): distance in world coordinates
        self.motion_tracking_global_per_step = []  # List of [num_envs, num_bodies, 3] tensors
        
        # Motion tracking (local frame): distance in robot anchor-relative coordinates
        self.motion_tracking_local_per_step = []  # List of [num_envs, num_bodies, 3] tensors
        
        # Track step count
        self.step_count = 0
        
    def collect_step_metrics(self):
        """Collect metrics for the current step."""
        env = self.env
        
        # 1. Energy: |torque * joint_vel|
        # Following the reference: torch.norm(torch.abs(torque * joint_vel), dim=-1)
        # Here we store per-joint to get detailed breakdown
        energy = torch.abs(env.torques * env.simulator.dof_vel)  # [num_envs, num_dofs]
        self.energy_per_step.append(energy.clone().cpu())
        
        # 2. Foot slippage: velocity when in contact with ground (same as reward)
        # is_contact = contact_force > 1.0, foot_planar_velocity = norm(velocity[:, :2])
        is_contact = torch.norm(env.simulator.contact_forces[:, env.feet_indices, :], dim=-1) > 1.0  # [num_envs, 2]
        foot_vel = env.simulator._rigid_body_vel[:, env.feet_indices, :2]  # [num_envs, 2, 2] (2 feet, xy vel)
        foot_planar_velocity = torch.linalg.norm(foot_vel, dim=-1)  # [num_envs, 2] in m/s
        
        # Slippage as velocity (same as reward function)
        slippage = is_contact * foot_planar_velocity  # [num_envs, 2] in m/s
        self.foot_slippage_left_per_step.append(slippage[:, 0].clone().cpu())
        self.foot_slippage_right_per_step.append(slippage[:, 1].clone().cpu())
        
        # 3. Foot contact force: magnitude of contact force
        foot_contact_forces = torch.norm(env.simulator.contact_forces[:, env.feet_indices, :], dim=-1)  # [num_envs, 2]
        self.foot_contact_force_left_per_step.append(foot_contact_forces[:, 0].clone().cpu())
        self.foot_contact_force_right_per_step.append(foot_contact_forces[:, 1].clone().cpu())
        
        # 4. Action rate: |action_t - action_t-1|
        if self.prev_actions is not None:
            action_rate = torch.abs(env.actions - self.prev_actions)  # [num_envs, num_dofs]
            self.action_rate_per_step.append(action_rate.clone().cpu())
        else:
            # First step: no previous action, so action_rate is 0
            action_rate = torch.zeros_like(env.actions)
            self.action_rate_per_step.append(action_rate.clone().cpu())
        
        # 5. Action smoothness (jerk): (action_t - 2*action_t-1 + action_t-2)^2
        if self.prev_actions is not None and self.prev_prev_actions is not None:
            action_acc = env.actions - 2 * self.prev_actions + self.prev_prev_actions  # [num_envs, num_dofs]
            action_smoothness = action_acc ** 2  # [num_envs, num_dofs]
            self.action_smoothness_per_step.append(action_smoothness.clone().cpu())
        else:
            # First two steps: no smoothness calculation
            action_smoothness = torch.zeros_like(env.actions)
            self.action_smoothness_per_step.append(action_smoothness.clone().cpu())
        
        # 6. Joint acceleration: |dof_vel_t - dof_vel_t-1| / dt
        if self.prev_dof_vel is not None:
            joint_acc = torch.abs(env.simulator.dof_vel - self.prev_dof_vel) / env.dt  # [num_envs, num_dofs]
            self.joint_acc_per_step.append(joint_acc.clone().cpu())
        else:
            joint_acc = torch.zeros_like(env.simulator.dof_vel)
            self.joint_acc_per_step.append(joint_acc.clone().cpu())
        self.prev_dof_vel = env.simulator.dof_vel.clone()
        
        # 7. Base angular velocity tracking error & Reference energy
        if hasattr(env, '_motion_lib'):
            motion_times = (env.episode_length_buf) * env.dt + env.motion_start_times
            motion_res = env._motion_lib.get_motion_state(env.motion_ids, motion_times, env.env_origins)
            ref_base_ang_vel = motion_res["root_ang_vel"]  # [num_envs, 3]
            actual_base_ang_vel = env.base_ang_vel  # [num_envs, 3]
            base_ang_vel_error = torch.norm(ref_base_ang_vel - actual_base_ang_vel, dim=-1)  # [num_envs]
            self.base_ang_vel_error_per_step.append(base_ang_vel_error.clone().cpu())
            
            # 8. Reference energy (PD controller approximation)
            # Estimate torque needed to track reference: τ ≈ Kp*(q_ref - q) + Kd*(dq_ref - dq)
            ref_dof_pos = motion_res["dof_pos"]  # [num_envs, num_dofs]
            ref_dof_vel = motion_res["dof_vel"]  # [num_envs, num_dofs]
            
            # Get Kp and Kd from env
            Kp = env.p_gains  # [num_dofs]
            Kd = env.d_gains  # [num_dofs]
            
            # Estimate reference tracking torque
            pos_error = ref_dof_pos - env.simulator.dof_pos
            vel_error = ref_dof_vel - env.simulator.dof_vel
            ref_torque_approx = Kp * pos_error + Kd * vel_error  # [num_envs, num_dofs]
            
            # Reference energy: |τ_ref × dq_ref|
            ref_energy = torch.abs(ref_torque_approx * ref_dof_vel)  # [num_envs, num_dofs]
            self.ref_energy_per_step.append(ref_energy.clone().cpu())
        
        # Update previous actions for next step
        self.prev_prev_actions = self.prev_actions
        self.prev_actions = env.actions.clone()
        
        # 5. Motion tracking (global frame)
        if hasattr(env, 'dif_global_body_pos'):
            motion_tracking_global = env.dif_global_body_pos.clone().cpu()  # [num_envs, num_bodies, 3]
            self.motion_tracking_global_per_step.append(motion_tracking_global)
        
        # 6. Motion tracking (local frame - robot anchor relative)
        if hasattr(env, 'dif_local_body_pos'):
            motion_tracking_local = env.dif_local_body_pos.clone().cpu()  # [num_envs, num_bodies, 3]
            self.motion_tracking_local_per_step.append(motion_tracking_local)
        
        self.step_count += 1
        
    def compute_summary_statistics(self):
        """Compute summary statistics from collected metrics."""
        results = {}
        
        if self.step_count == 0:
            logger.warning("No steps collected!")
            return results
            
        # ===================== 1. Energy =====================
        if len(self.energy_per_step) > 0:
            # Stack: [num_steps, num_envs, num_dofs]
            energy_stack = torch.stack(self.energy_per_step, dim=0)
            
            # Mean energy per step (average over all steps and envs)
            mean_energy_per_step = energy_stack.mean(dim=0).mean(dim=0)  # [num_dofs]
            
            # Average energy (mean over all joints)
            avg_energy = mean_energy_per_step.mean()
            
            results['energy'] = {
                'per_joint': {self.dof_names[i]: mean_energy_per_step[i].item() for i in range(len(self.dof_names))},
                'average': avg_energy.item(),
            }
            
        # ===================== 2. Foot Slippage =====================
        if len(self.foot_slippage_left_per_step) > 0:
            left_slip = torch.stack(self.foot_slippage_left_per_step, dim=0)  # [num_steps, num_envs]
            right_slip = torch.stack(self.foot_slippage_right_per_step, dim=0)
            
            # Sum over trajectory
            left_slip_sum = left_slip.sum(dim=0).mean()  # mean over envs
            right_slip_sum = right_slip.sum(dim=0).mean()
            avg_slip = (left_slip_sum + right_slip_sum) / 2
            
            results['foot_slippage'] = {
                'left': left_slip_sum.item(),
                'right': right_slip_sum.item(),
                'average': avg_slip.item(),
            }
            
        # ===================== 3. Foot Contact Force =====================
        if len(self.foot_contact_force_left_per_step) > 0:
            left_force = torch.stack(self.foot_contact_force_left_per_step, dim=0)  # [num_steps, num_envs]
            right_force = torch.stack(self.foot_contact_force_right_per_step, dim=0)
            
            # Mean over trajectory (average contact force per step)
            left_force_mean = left_force.mean(dim=0).mean()  # mean over envs
            right_force_mean = right_force.mean(dim=0).mean()
            avg_force = (left_force_mean + right_force_mean) / 2
            
            # Max force during trajectory
            left_force_max = left_force.max(dim=0)[0].mean()
            right_force_max = right_force.max(dim=0)[0].mean()
            
            # Standard deviation (variability of contact force)
            left_force_std = left_force.std(dim=0).mean()  # std over time, mean over envs
            right_force_std = right_force.std(dim=0).mean()
            avg_force_std = (left_force_std + right_force_std) / 2
            
            results['foot_contact_force'] = {
                'left_mean': left_force_mean.item(),
                'right_mean': right_force_mean.item(),
                'average_mean': avg_force.item(),
                'left_max': left_force_max.item(),
                'right_max': right_force_max.item(),
                'left_std': left_force_std.item(),
                'right_std': right_force_std.item(),
                'average_std': avg_force_std.item(),
            }
            
        # ===================== 4. Action Rate =====================
        if len(self.action_rate_per_step) > 1:  # Need at least 2 steps
            action_rate_stack = torch.stack(self.action_rate_per_step, dim=0)  # [num_steps, num_envs, num_dofs]
            
            # Skip first step (action_rate is 0 for first step) and compute mean
            per_joint_action_rate = action_rate_stack[1:].mean(dim=0).mean(dim=0)  # [num_dofs]
            avg_action_rate = per_joint_action_rate.mean()
            
            results['action_rate'] = {
                'per_joint': {self.dof_names[i]: per_joint_action_rate[i].item() for i in range(len(self.dof_names))},
                'average': avg_action_rate.item(),
            }
            
        # ===================== 5. Action Smoothness (Jerk) =====================
        if len(self.action_smoothness_per_step) > 2:  # Need at least 3 steps
            smoothness_stack = torch.stack(self.action_smoothness_per_step, dim=0)  # [num_steps, num_envs, num_dofs]
            
            # Skip first two steps and compute mean
            per_joint_smoothness = smoothness_stack[2:].mean(dim=0).mean(dim=0)  # [num_dofs]
            mean_smoothness = per_joint_smoothness.mean()
            
            results['action_smoothness'] = {
                'per_joint': {self.dof_names[i]: per_joint_smoothness[i].item() for i in range(len(self.dof_names))},
                'average': mean_smoothness.item(),
            }
            
        # ===================== 6. Joint Acceleration =====================
        if len(self.joint_acc_per_step) > 1:
            joint_acc_stack = torch.stack(self.joint_acc_per_step, dim=0)  # [num_steps, num_envs, num_dofs]
            
            per_joint_acc = joint_acc_stack[1:].mean(dim=0).mean(dim=0)  # [num_dofs]
            mean_joint_acc = per_joint_acc.mean()
            
            results['joint_acceleration'] = {
                'per_joint': {self.dof_names[i]: per_joint_acc[i].item() for i in range(len(self.dof_names))},
                'average': mean_joint_acc.item(),
            }
            
        # ===================== 7. Base Angular Velocity Error =====================
        if len(self.base_ang_vel_error_per_step) > 0:
            base_ang_vel_error_stack = torch.stack(self.base_ang_vel_error_per_step, dim=0)  # [num_steps, num_envs]
            mean_base_ang_vel_error = base_ang_vel_error_stack.mean()
            
            results['base_ang_vel_error'] = {
                'average': mean_base_ang_vel_error.item(),
            }
            
        # ===================== 8. Reference Energy (PD Approximation) =====================
        if len(self.ref_energy_per_step) > 0:
            ref_energy_stack = torch.stack(self.ref_energy_per_step, dim=0)  # [num_steps, num_envs, num_dofs]
            
            # Mean reference energy per joint
            per_joint_ref_energy = ref_energy_stack.mean(dim=0).mean(dim=0)  # [num_dofs]
            mean_ref_energy = per_joint_ref_energy.mean()
            
            results['ref_energy'] = {
                'per_joint': {self.dof_names[i]: per_joint_ref_energy[i].item() for i in range(len(self.dof_names))},
                'average': mean_ref_energy.item(),
            }
            
        # ===================== 9. Motion Tracking (Global Frame) =====================
        if len(self.motion_tracking_global_per_step) > 0:
            # Stack: [num_steps, num_envs, num_bodies, 3]
            motion_stack = torch.stack(self.motion_tracking_global_per_step, dim=0)
            
            # Compute distance (L2 norm) per body
            motion_dist = torch.norm(motion_stack, dim=-1)  # [num_steps, num_envs, num_bodies]
            
            # Statistics over trajectory (per body)
            # Mean over steps and envs
            per_body_mean = motion_dist.mean(dim=0).mean(dim=0)  # [num_bodies]
            per_body_min = motion_dist.min(dim=0)[0].mean(dim=0)  # [num_bodies]
            per_body_max = motion_dist.max(dim=0)[0].mean(dim=0)  # [num_bodies]
            
            # Overall statistics
            overall_mean = per_body_mean.mean()
            overall_min = per_body_min.min()
            overall_max = per_body_max.max()
            
            # Per-body results
            num_bodies = len(self.body_names) if len(self.body_names) > 0 else per_body_mean.shape[0]
            per_body_results = {}
            for i in range(per_body_mean.shape[0]):
                body_name = self.body_names[i] if i < len(self.body_names) else f"body_{i}"
                per_body_results[body_name] = {
                    'mean': per_body_mean[i].item(),
                    'min': per_body_min[i].item(),
                    'max': per_body_max[i].item(),
                }
            
            results['motion_tracking_global'] = {
                'per_body': per_body_results,
                'overall': {
                    'mean': overall_mean.item(),
                    'min': overall_min.item(),
                    'max': overall_max.item(),
                },
            }
            
        # ===================== 10. Motion Tracking (Local Frame - Robot Anchor Relative) =====================
        if len(self.motion_tracking_local_per_step) > 0:
            # Stack: [num_steps, num_envs, num_bodies, 3]
            motion_local_stack = torch.stack(self.motion_tracking_local_per_step, dim=0)
            
            # Compute distance (L2 norm) per body
            motion_local_dist = torch.norm(motion_local_stack, dim=-1)  # [num_steps, num_envs, num_bodies]
            
            # Statistics
            per_body_local_mean = motion_local_dist.mean(dim=0).mean(dim=0)  # [num_bodies]
            per_body_local_min = motion_local_dist.min(dim=0)[0].mean(dim=0)
            per_body_local_max = motion_local_dist.max(dim=0)[0].mean(dim=0)
            
            # Overall statistics
            overall_local_mean = per_body_local_mean.mean()
            overall_local_min = per_body_local_min.min()
            overall_local_max = per_body_local_max.max()
            
            # Per-body results
            per_body_local_results = {}
            for i in range(per_body_local_mean.shape[0]):
                body_name = self.body_names[i] if i < len(self.body_names) else f"body_{i}"
                per_body_local_results[body_name] = {
                    'mean': per_body_local_mean[i].item(),
                    'min': per_body_local_min[i].item(),
                    'max': per_body_local_max[i].item(),
                }
            
            results['motion_tracking_local'] = {
                'per_body': per_body_local_results,
                'overall': {
                    'mean': overall_local_mean.item(),
                    'min': overall_local_min.item(),
                    'max': overall_local_max.item(),
                },
            }
            
        results['metadata'] = {
            'num_steps': self.step_count,
            'num_envs': self.num_envs,
            'num_dofs': self.num_dofs,
        }
        
        return results
    
    def save_results(self, save_path: Path, results: dict):
        """Save results to a text file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAJECTORY EVALUATION METRICS\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            if 'metadata' in results:
                f.write("METADATA\n")
                f.write("-" * 40 + "\n")
                for key, value in results['metadata'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # 1. Energy
            if 'energy' in results:
                f.write("1. ENERGY (|torque * joint_velocity| - mean per step)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average energy across all joints: {results['energy']['average']:.4f}\n")
                f.write("\n  Per-joint mean energy (averaged over steps):\n")
                for joint_name, energy in results['energy']['per_joint'].items():
                    f.write(f"    {joint_name}: {energy:.4f}\n")
                f.write("\n")
            
            # 2. Foot Slippage
            if 'foot_slippage' in results:
                f.write("2. FOOT SLIPPAGE (velocity when in contact)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Left foot (sum over trajectory): {results['foot_slippage']['left']:.4f}\n")
                f.write(f"  Right foot (sum over trajectory): {results['foot_slippage']['right']:.4f}\n")
                f.write(f"  Average: {results['foot_slippage']['average']:.4f}\n")
                f.write("\n")
            
            # 3. Foot Contact Force
            if 'foot_contact_force' in results:
                f.write("3. FOOT CONTACT FORCE\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Left foot (mean): {results['foot_contact_force']['left_mean']:.4f}\n")
                f.write(f"  Right foot (mean): {results['foot_contact_force']['right_mean']:.4f}\n")
                f.write(f"  Average (mean): {results['foot_contact_force']['average_mean']:.4f}\n")
                f.write(f"  Left foot (max): {results['foot_contact_force']['left_max']:.4f}\n")
                f.write(f"  Right foot (max): {results['foot_contact_force']['right_max']:.4f}\n")
                f.write(f"  Left foot (std): {results['foot_contact_force']['left_std']:.4f}\n")
                f.write(f"  Right foot (std): {results['foot_contact_force']['right_std']:.4f}\n")
                f.write(f"  Average (std): {results['foot_contact_force']['average_std']:.4f}\n")
                f.write("\n")
            
            # 4. Action Rate
            if 'action_rate' in results:
                f.write("4. ACTION RATE (|action_t - action_t-1|)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average action rate: {results['action_rate']['average']:.6f}\n")
                f.write("\n  Per-joint action rate:\n")
                for joint_name, rate in results['action_rate']['per_joint'].items():
                    f.write(f"    {joint_name}: {rate:.6f}\n")
                f.write("\n")
            
            # 5. Action Smoothness (Jerk)
            if 'action_smoothness' in results:
                f.write("5. ACTION SMOOTHNESS - Jerk ((action_t - 2*action_t-1 + action_t-2)^2)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average smoothness (lower is better): {results['action_smoothness']['average']:.6f}\n")
                f.write("\n  Per-joint smoothness:\n")
                for joint_name, smoothness in results['action_smoothness']['per_joint'].items():
                    f.write(f"    {joint_name}: {smoothness:.6f}\n")
                f.write("\n")
            
            # 6. Joint Acceleration
            if 'joint_acceleration' in results:
                f.write("6. JOINT ACCELERATION (|dof_vel_t - dof_vel_t-1| / dt)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average joint acceleration: {results['joint_acceleration']['average']:.4f} rad/s^2\n")
                f.write("\n  Per-joint acceleration:\n")
                for joint_name, acc in results['joint_acceleration']['per_joint'].items():
                    f.write(f"    {joint_name}: {acc:.4f}\n")
                f.write("\n")
            
            # 7. Base Angular Velocity Error
            if 'base_ang_vel_error' in results:
                f.write("7. BASE ANGULAR VELOCITY TRACKING ERROR\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average error (L2 norm): {results['base_ang_vel_error']['average']:.6f} rad/s\n")
                f.write("\n")
            
            # 8. Reference Energy (PD Approximation)
            if 'ref_energy' in results:
                f.write("8. REFERENCE ENERGY (PD Controller Approximation)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average reference energy: {results['ref_energy']['average']:.4f}\n")
                f.write("  Note: Estimated using torque_ref ~= Kp*(q_ref-q) + Kd*(dq_ref-dq)\n")
                f.write("\n  Per-joint reference energy:\n")
                for joint_name, energy in results['ref_energy']['per_joint'].items():
                    f.write(f"    {joint_name}: {energy:.4f}\n")
                f.write("\n")
            
            # 9. Motion Tracking (Global)
            if 'motion_tracking_global' in results:
                f.write("9. MOTION TRACKING - GLOBAL FRAME (world coordinates)\n")
                f.write("-" * 40 + "\n")
                overall = results['motion_tracking_global']['overall']
                f.write(f"  Overall mean distance: {overall['mean']:.6f} m\n")
                f.write(f"  Overall min distance: {overall['min']:.6f} m\n")
                f.write(f"  Overall max distance: {overall['max']:.6f} m\n")
                f.write("\n  Per-body statistics:\n")
                for body_name, stats in results['motion_tracking_global']['per_body'].items():
                    f.write(f"    {body_name}:\n")
                    f.write(f"      mean: {stats['mean']:.6f} m, min: {stats['min']:.6f} m, max: {stats['max']:.6f} m\n")
                f.write("\n")
            
            # 10. Motion Tracking (Local)
            if 'motion_tracking_local' in results:
                f.write("10. MOTION TRACKING - LOCAL FRAME (robot anchor relative)\n")
                f.write("-" * 40 + "\n")
                overall_local = results['motion_tracking_local']['overall']
                f.write(f"  Overall mean distance: {overall_local['mean']:.6f} m\n")
                f.write(f"  Overall min distance: {overall_local['min']:.6f} m\n")
                f.write(f"  Overall max distance: {overall_local['max']:.6f} m\n")
                f.write("\n  Per-body statistics:\n")
                for body_name, stats in results['motion_tracking_local']['per_body'].items():
                    f.write(f"    {body_name}:\n")
                    f.write(f"      mean: {stats['mean']:.6f} m, min: {stats['min']:.6f} m, max: {stats['max']:.6f} m\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Results saved to {save_path}")


def evaluate_with_metrics(algo, env, num_steps, metrics_collector):
    """Run evaluation while collecting metrics.
    
    Uses algo's own evaluation logic to properly handle different policy types
    (standard PPO, PPO with motion_encoder, MHPPO with history/priv, etc.)
    """
    
    # Set environment to evaluation mode
    env.set_is_evaluating()
    
    # Reset environment
    obs_dict = env.reset_all()
    
    for key in obs_dict:
        obs_dict[key] = obs_dict[key].to(algo.device)
    
    # Get inference policy from algo (handles all complexity internally)
    eval_policy = algo._get_inference_policy()
    
    # Set eval_policy as algo attribute (required by _pre_eval_env_step)
    algo.eval_policy = eval_policy
    
    logger.info(f"Starting evaluation for {num_steps} steps...")
    
    # Initialize actor_state with proper structure
    init_actions = torch.zeros(env.num_envs, env.num_dof, device=algo.device)
    actor_state = {"obs": obs_dict, "actions": init_actions, "step": 0}
    
    for step in range(num_steps):
        with torch.inference_mode():
            # Use algo's pre_eval_env_step if available (handles all edge cases)
            if hasattr(algo, '_pre_eval_env_step'):
                actor_state["step"] = step
                actor_state["obs"] = obs_dict
                actor_state = algo._pre_eval_env_step(actor_state)
                actions = actor_state["actions"]
            else:
                # Fallback: simple evaluation
                if 'actor_obs' in obs_dict:
                    actions = eval_policy(obs_dict['actor_obs'])
                else:
                    actions = eval_policy(obs_dict)
        
        # Step the environment
        actor_state_for_env = {"actions": actions}
        obs_dict, rewards, dones, infos = env.step(actor_state_for_env)
        
        for key in obs_dict:
            obs_dict[key] = obs_dict[key].to(algo.device)
        
        # Collect metrics AFTER the step
        metrics_collector.collect_step_metrics()
        
        if step % 100 == 0:
            logger.info(f"Step {step}/{num_steps}")
        
        # Check if motion ended (using time_out_buf only - keep it simple!)
        if hasattr(env, 'time_out_buf') and env.time_out_buf.any():
            # For single env evaluation
            if env.num_envs == 1 and env.time_out_buf[0]:
                logger.info(f"Motion ended at step {step+1}")
                break
            # For multi-env
            elif env.time_out_buf.all():
                logger.info(f"All motions ended at step {step+1}")
                break
    
    logger.info(f"Evaluation completed. Total steps: {metrics_collector.step_count}")


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval_metrics.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config
            
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    if simulator_type == 'IsaacGym':
        import isaacgym
        
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config

    pre_process_config(config)

    # use config.device if specified, otherwise use cuda if available
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    ckpt_num = config.checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
    config.num_envs = 1
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    config.env.config.ckpt_dir = str(checkpoint.parent)
    
    # Get number of evaluation steps
    num_eval_steps = config.get("num_eval_steps", 201)
    
    env = instantiate(config.env, device=device)

    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    # Initialize metrics collector
    metrics_collector = MetricsCollector(env, device)
    
    # Run evaluation with metrics collection
    evaluate_with_metrics(algo, env, num_eval_steps, metrics_collector)
    
    # Compute summary statistics
    results = metrics_collector.compute_summary_statistics()
    
    # Save results
    # Check if custom save path is provided
    if config.get("metrics_save_path", None):
        save_path = Path(config.metrics_save_path)
    else:
        # Default: save to checkpoint directory
        metrics_dir = checkpoint.parent / "metrics"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = metrics_dir / f"trajectory_metrics_ckpt_{ckpt_num}_{timestamp}.txt"
    
    metrics_collector.save_results(save_path, results)
    
    # Print summary to console
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    if 'energy' in results:
        logger.info(f"Energy - Average per step: {results['energy']['average']:.4f}")
    
    if 'foot_slippage' in results:
        logger.info(f"Foot Slippage - Left: {results['foot_slippage']['left']:.4f}, Right: {results['foot_slippage']['right']:.4f}, Avg: {results['foot_slippage']['average']:.4f}")
    
    if 'foot_contact_force' in results:
        logger.info(f"Foot Contact Force - Left: {results['foot_contact_force']['left_mean']:.4f}, Right: {results['foot_contact_force']['right_mean']:.4f}, Avg: {results['foot_contact_force']['average_mean']:.4f}")
    
    if 'action_rate' in results:
        logger.info(f"Action Rate - Average: {results['action_rate']['average']:.6f}")
    
    if 'motion_tracking' in results:
        overall = results['motion_tracking']['overall']
        logger.info(f"Motion Tracking - Mean: {overall['mean']:.6f}m, Min: {overall['min']:.6f}m, Max: {overall['max']:.6f}m")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
