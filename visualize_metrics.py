#!/usr/bin/env python3
"""
Visualize ablation study metrics with bar charts
Automatically reads data from ablation_results/*.txt files
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def parse_metrics_file(filepath):
    """Parse a metrics txt file and extract key values"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Average energy
    match = re.search(r'Average energy across all joints:\s+([\d.]+)', content)
    if match:
        metrics['energy_avg'] = float(match.group(1))
    
    # Foot slippage
    match_left = re.search(r'Left foot \(sum over trajectory\):\s+([\d.]+)', content)
    match_right = re.search(r'Right foot \(sum over trajectory\):\s+([\d.]+)', content)
    if match_left and match_right:
        metrics['slippage_avg'] = (float(match_left.group(1)) + float(match_right.group(1))) / 2
    
    # Contact force mean
    matches = re.findall(r'(Left|Right) foot \(mean\):\s+([\d.]+)', content)
    if len(matches) >= 2:
        forces = [float(val) for _, val in matches]
        metrics['contact_force_avg'] = sum(forces) / len(forces)
    
    # Contact force std
    matches_std = re.findall(r'(Left|Right) foot \(std\):\s+([\d.]+)', content)
    if len(matches_std) >= 2:
        forces_std = [float(val) for _, val in matches_std]
        metrics['contact_force_std'] = sum(forces_std) / len(forces_std)
    
    # Action rate
    match = re.search(r'Average action rate:\s+([\d.]+)', content)
    if match:
        metrics['action_rate'] = float(match.group(1))
    
    # Motion tracking max (global)
    match = re.search(r'GLOBAL FRAME.*?Overall max distance:\s+([\d.]+)\s+m', content, re.DOTALL)
    if match:
        metrics['motion_tracking_global_max'] = float(match.group(1))
    else:
        # Fallback to old format (without frame specification)
        match = re.search(r'Overall max distance:\s+([\d.]+)\s+m', content)
        if match:
            metrics['motion_tracking_global_max'] = float(match.group(1))
    
    # Motion tracking max (local)
    match = re.search(r'LOCAL FRAME.*?Overall max distance:\s+([\d.]+)\s+m', content, re.DOTALL)
    if match:
        metrics['motion_tracking_local_max'] = float(match.group(1))
    
    # Action smoothness
    match = re.search(r'Average smoothness \(lower is better\):\s+([\d.]+)', content)
    if match:
        metrics['action_smoothness'] = float(match.group(1))
    
    # Joint acceleration
    match = re.search(r'Average joint acceleration:\s+([\d.]+)', content)
    if match:
        metrics['joint_acceleration'] = float(match.group(1))
    
    # Base angular velocity error
    match = re.search(r'Average error \(L2 norm\):\s+([\d.]+)\s+rad/s', content)
    if match:
        metrics['base_ang_vel_error'] = float(match.group(1))
    
    return metrics

results_dir = Path('/home/kyungminlee/PBHC/ablation_results')

def load_models(model_order):
    models_data = []
    model_labels = []
    
    for filename, label in model_order:
        filepath = results_dir / filename
        if filepath.exists():
            metrics = parse_metrics_file(filepath)
            if metrics:
                models_data.append(metrics)
                model_labels.append(label)
        else:
            print(f"⚠️  File not found: {filename}")
    
    return models_data, model_labels

def create_comparison_plot(models_data, model_labels, title_suffix, filename_suffix):
    """Create comprehensive comparison plot"""
    
    # Extract metrics
    energy_avg = [m.get('energy_avg', 0) for m in models_data]
    slippage_avg = [m.get('slippage_avg', 0) for m in models_data]
    contact_force_avg = [m.get('contact_force_avg', 0) for m in models_data]
    contact_force_std = [m.get('contact_force_std', 0) for m in models_data]
    action_rate = [m.get('action_rate', 0) for m in models_data]
    action_smoothness = [m.get('action_smoothness', 0) for m in models_data]
    joint_acceleration = [m.get('joint_acceleration', 0) for m in models_data]
    base_ang_vel_error = [m.get('base_ang_vel_error', 0) for m in models_data]
    motion_tracking_global_max = [m.get('motion_tracking_global_max', 0) for m in models_data]
    motion_tracking_local_max = [m.get('motion_tracking_local_max', 0) for m in models_data]
    
    # Create figure (4x3 grid for 10 plots)
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    fig.suptitle(f'Motion Tracking Evaluation: {title_suffix}', fontsize=16, fontweight='bold', y=0.995)
    
    n_models = len(models_data)
    model_colors = plt.cm.Set3(np.linspace(0, 1, max(n_models, 8)))[:n_models]
    
    # 1. Energy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(n_models), energy_avg, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Energy (Nm·rad/s)', fontsize=10)
    ax1.set_title('1. Average Energy per Step', fontweight='bold', fontsize=11)
    ax1.set_xticks(range(n_models))
    ax1.set_xticklabels(model_labels, fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Foot Slippage
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(n_models), slippage_avg, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Slippage (m)', fontsize=10)
    ax2.set_title('2. Foot Slippage', fontweight='bold', fontsize=11)
    ax2.set_xticks(range(n_models))
    ax2.set_xticklabels(model_labels, fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Motion Tracking - Global Frame
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(n_models), motion_tracking_global_max, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Max Distance (m)', fontsize=10)
    ax3.set_title('3. Motion Tracking (Global)', fontweight='bold', fontsize=11)
    ax3.set_xticks(range(n_models))
    ax3.set_xticklabels(model_labels, fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Motion Tracking - Local Frame
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(range(n_models), motion_tracking_local_max, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_ylabel('Max Distance (m)', fontsize=10)
    ax4.set_title('4. Motion Tracking (Local)', fontweight='bold', fontsize=11)
    ax4.set_xticks(range(n_models))
    ax4.set_xticklabels(model_labels, fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Contact Force (Mean & Std)
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(n_models)
    width = 0.35
    ax5.bar(x - width/2, contact_force_avg, width, label='Mean', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1)
    ax5.bar(x + width/2, contact_force_std, width, label='Std', color='#F39C12', alpha=0.8, edgecolor='black', linewidth=1)
    ax5.set_ylabel('Force (N)', fontsize=10)
    ax5.set_title('5. Contact Force (Mean & Std)', fontweight='bold', fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels(model_labels, fontsize=8)
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Base Angular Velocity Error
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(range(n_models), base_ang_vel_error, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax6.set_ylabel('Error (rad/s)', fontsize=10)
    ax6.set_title('6. Base Ang Vel Error', fontweight='bold', fontsize=11)
    ax6.set_xticks(range(n_models))
    ax6.set_xticklabels(model_labels, fontsize=8)
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Action Rate
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.bar(range(n_models), action_rate, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax7.set_ylabel('Action Rate', fontsize=10)
    ax7.set_title('7. Action Rate', fontweight='bold', fontsize=11)
    ax7.set_xticks(range(n_models))
    ax7.set_xticklabels(model_labels, fontsize=8)
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Action Smoothness
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.bar(range(n_models), action_smoothness, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax8.set_ylabel('Smoothness (jerk²)', fontsize=10)
    ax8.set_title('8. Action Smoothness', fontweight='bold', fontsize=11)
    ax8.set_xticks(range(n_models))
    ax8.set_xticklabels(model_labels, fontsize=8)
    ax8.grid(axis='y', alpha=0.3)
    
    # 9. Joint Acceleration
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.bar(range(n_models), joint_acceleration, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax9.set_ylabel('Accel (rad/s²)', fontsize=10)
    ax9.set_title('9. Joint Acceleration', fontweight='bold', fontsize=11)
    ax9.set_xticks(range(n_models))
    ax9.set_xticklabels(model_labels, fontsize=8)
    ax9.grid(axis='y', alpha=0.3)
    
    # 10. Overall Score (Absolute Thresholds)
    ax10 = fig.add_subplot(gs[3, :])  # Full width of row 4
    
    def score_absolute(value, excellent, poor):
        """Score based on absolute thresholds (lower is better)
        excellent: threshold for 100 points
        poor: threshold for 0 points
        """
        if value <= 0:
            return 0
        if value <= excellent:
            return 100
        elif value >= poor:
            return 0
        else:
            # Linear interpolation between excellent and poor
            return 100 * (poor - value) / (poor - excellent)
    
    # Define absolute thresholds for each metric
    overall_score = []
    for i in range(n_models):
        scores = []
        
        # Motion tracking (global): excellent < 0.15m, poor > 0.35m
        if motion_tracking_global_max[i] > 0:
            scores.append(score_absolute(motion_tracking_global_max[i], 0.15, 0.35))
        
        # Motion tracking (local): excellent < 0.15m, poor > 0.35m
        if motion_tracking_local_max[i] > 0:
            scores.append(score_absolute(motion_tracking_local_max[i], 0.15, 0.35))
        
        # Foot slippage: excellent < 5, poor > 20 (sum of m/s over trajectory)
        if slippage_avg[i] > 0:
            scores.append(score_absolute(slippage_avg[i], 5, 20))
        
        # Action smoothness: excellent < 0.0001, poor > 0.001
        if action_smoothness[i] > 0:
            scores.append(score_absolute(action_smoothness[i], 0.0001, 0.001))
        
        # Contact force std: excellent < 20, poor > 60
        if contact_force_std[i] > 0:
            scores.append(score_absolute(contact_force_std[i], 20, 60))
        
        # Joint acceleration: excellent < 10, poor > 40
        if joint_acceleration[i] > 0:
            scores.append(score_absolute(joint_acceleration[i], 10, 40))
        
        # Base ang vel error: excellent < 0.01, poor > 0.1
        if base_ang_vel_error[i] > 0:
            scores.append(score_absolute(base_ang_vel_error[i], 0.01, 0.1))
        
        # Energy: excellent < 4, poor > 10
        if energy_avg[i] > 0:
            scores.append(score_absolute(energy_avg[i], 4, 10))
        
        # Action rate: excellent < 0.05, poor > 0.2
        if action_rate[i] > 0:
            scores.append(score_absolute(action_rate[i], 0.05, 0.2))
        
        # Average of all scores
        avg_score = np.mean(scores) if scores else 0
        overall_score.append(avg_score)
    
    overall_score = np.array(overall_score)
    
    ax10.bar(range(n_models), overall_score, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax10.set_ylabel('Score (0-100)', fontsize=11, fontweight='bold')
    ax10.set_title('10. Overall Score (Average of Normalized Metrics)', 
                   fontweight='bold', fontsize=11)
    ax10.set_xticks(range(n_models))
    ax10.set_xticklabels(model_labels, fontsize=9)
    ax10.set_ylim([0, 105])
    ax10.grid(axis='y', alpha=0.3)
    
    for i, val in enumerate(overall_score):
        ax10.text(i, val + 2, f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Save
    output_dir = Path('/home/kyungminlee/PBHC/ablation_results')
    plt.savefig(output_dir / f'metrics_comparison_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'metrics_comparison_{filename_suffix}.pdf', bbox_inches='tight')
    
    return overall_score, model_labels

# Define model orders
comparison1_order = [
    ('gmt_teacher.txt', 'KungfuBot2\nTeacher'),
    ('gmt_student.txt', 'KungfuBot2\nStudent'),
    ('holosoma.txt', 'Holosoma'),
    ('default.txt', 'KungfuBot1'),
]

comparison2_order = [
    ('default.txt', 'Baseline'),
    ('default_history10.txt', 'History\n4→10'),
    ('default_revise_ref_info.txt', 'Phase→\nRef Info'),
    ('default_add_future_motion.txt', '+ Future'),
    ('default_add_priv10.txt', '+ Priv'),
    ('default_gmt_reward.txt', '+ Reward'),
    ('default_robot_control.txt', '+ Robot'),
]

# Generate comparisons
print("\n" + "="*80)
print("COMPARISON 1: Main Methods")
print("="*80)

models_data1, model_labels1 = load_models(comparison1_order)
print(f"✅ Loaded {len(models_data1)} models")
overall_score1, labels1 = create_comparison_plot(models_data1, model_labels1, 
                                                   'Main Methods', 'main_methods')

print("\n✅ Saved: ablation_results/metrics_comparison_main_methods.png/pdf")
rankings1 = sorted(zip(labels1, overall_score1), key=lambda x: x[1], reverse=True)
for i, (model, score) in enumerate(rankings1, 1):
    print(f"{i}. {model.replace(chr(10), ' '):20s}: {score:5.1f}/100")

print("\n" + "="*80)
print("COMPARISON 2: KungfuBot1 Ablations")
print("="*80)

models_data2, model_labels2 = load_models(comparison2_order)
print(f"✅ Loaded {len(models_data2)} models")
overall_score2, labels2 = create_comparison_plot(models_data2, model_labels2,
                                                   'KungfuBot1 Ablations', 'kungfubot1_ablations')

print("\n✅ Saved: ablation_results/metrics_comparison_kungfubot1_ablations.png/pdf")
rankings2 = sorted(zip(labels2, overall_score2), key=lambda x: x[1], reverse=True)
for i, (model, score) in enumerate(rankings2, 1):
    print(f"{i}. {model.replace(chr(10), ' '):20s}: {score:5.1f}/100")

print("\n" + "="*80)
print("COMPLETE! 📊")
print("="*80)

plt.show()
