#!/usr/bin/env python3
"""
Visualize ablation study metrics with bar charts
Automatically reads data from ablation_results/*.txt files
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import colorsys


def _draw_wavy_cap(ax, x_center, y_cap, bar_width=0.7, amplitude_frac=0.03):
    """Draw a small wavy/zigzag line at y_cap to indicate truncated bar."""
    x_left = x_center - bar_width / 2
    x_right = x_center + bar_width / 2
    amp = abs(y_cap * amplitude_frac) if y_cap != 0 else 0.01
    n_pts = 8
    xx = np.linspace(x_left, x_right, n_pts)
    yy = y_cap + amp * np.sin(np.linspace(0, 4 * np.pi, n_pts))
    ax.plot(xx, yy, color='black', linewidth=1.2, zorder=5)


def _bar_chart_with_cap(ax, values, colors, y_max=None, value_fmt='.2g', bar_width=0.7):
    """Draw bars; only cap when some value exceeds y_max; else auto scale. Wavy + value when truncated."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    x = np.arange(n)
    finite = np.isfinite(values) & (values >= 0)
    if not np.any(finite):
        ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        return
    data_max = float(np.nanmax(values[finite]))
    # Apply cap only when there is an outlier; otherwise auto scale
    if y_max is not None and data_max > y_max:
        display_max = float(y_max)
    else:
        display_max = max(data_max * 1.15, 1e-9)
    label_margin = 0.02 * display_max if display_max > 0 else 0.01
    displayed = np.minimum(values, display_max)
    ax.bar(x, displayed, bar_width, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    for i in range(n):
        v = values[i]
        if not np.isfinite(v):
            continue
        h = min(v, display_max)
        if v > display_max:
            _draw_wavy_cap(ax, i, display_max, bar_width=bar_width)
            ax.text(i, display_max + label_margin, f'{v:{value_fmt}}', ha='center', va='bottom',
                    fontsize=7, rotation=90 if v > display_max * 5 else 0)
        else:
            ax.text(i, h + label_margin, f'{v:{value_fmt}}', ha='center', va='bottom', fontsize=7)
    ax.set_ylim(0, display_max * 1.25)
    ax.grid(axis='y', alpha=0.3)


def _bar_chart_with_cap_signed(ax, values, colors, y_min=None, y_max=105, value_fmt='.1f', bar_width=0.7):
    """Overall score: only cap when value is out of range; else auto scale. Wavy + value when truncated."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    x = np.arange(n)
    data_min, data_max = float(np.nanmin(values)), float(np.nanmax(values))
    # Only use fixed cap when there is an outlier
    use_min_cap = y_min is not None and data_min < y_min
    use_max_cap = data_max > y_max
    display_min = y_min if use_min_cap else min(0, data_min - 20)
    display_max = y_max if use_max_cap else min(105, data_max + 10)
    displayed = np.clip(values, display_min, display_max)
    ax.bar(x, displayed, bar_width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for i in range(n):
        v = values[i]
        if not np.isfinite(v):
            continue
        h = displayed[i]
        if v < display_min:
            _draw_wavy_cap(ax, i, display_min, bar_width=bar_width)
            ax.text(i, display_min - 5, f'{v:{value_fmt}}', ha='center', va='top', fontsize=8, fontweight='bold')
        elif v > display_max:
            _draw_wavy_cap(ax, i, display_max, bar_width=bar_width)
            ax.text(i, display_max + 2, f'{v:{value_fmt}}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            offset = 2 if v >= 0 else -2
            va = 'bottom' if v >= 0 else 'top'
            ax.text(i, v + offset, f'{v:{value_fmt}}', ha='center', va=va, fontsize=8, fontweight='bold')
    ax.set_ylim(display_min - 15, display_max + 10)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)


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
    
    # Contact force mean (left, right, average, and imbalance)
    matches = re.findall(r'(Left|Right) foot \(mean\):\s+([\d.]+)', content)
    if len(matches) >= 2:
        left_mean = right_mean = None
        for label, val in matches:
            if label == 'Left':
                left_mean = float(val)
            else:
                right_mean = float(val)
        if left_mean is not None and right_mean is not None:
            metrics['contact_force_left_mean'] = left_mean
            metrics['contact_force_right_mean'] = right_mean
            metrics['contact_force_avg'] = (left_mean + right_mean) / 2
            metrics['contact_force_imbalance'] = abs(left_mean - right_mean)
    
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

# Baseline color mapping (fixed colors)
BASELINE_COLORS = {
    'kungfubot1': '#3498DB',        # Blue
    'kungfubot2': '#2ECC71',        # Green
    'holosoma': '#E67E22',          # Orange
    'maskedmimic_teacher': '#9B59B6',  # Purple
}

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert RGB tuple (0-1 range) to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def adjust_saturation(hex_color, saturation_factor=1.3):
    """Adjust saturation of a hex color (factor > 1 increases, < 1 decreases)"""
    rgb = hex_to_rgb(hex_color)
    hsv = colorsys.rgb_to_hsv(*rgb)
    # Increase saturation
    new_s = min(1.0, hsv[1] * saturation_factor)
    new_rgb = colorsys.hsv_to_rgb(hsv[0], new_s, hsv[2])
    return rgb_to_hex(new_rgb)

def detect_baseline(filename):
    """Detect which baseline a file belongs to"""
    filename_lower = filename.lower()
    # Check for explicit baseline names first
    if filename_lower == 'kungfubot1.txt':
        return 'kungfubot1'
    elif 'kungfubot2' in filename_lower and 'teacher' in filename_lower and 'no_hands' in filename_lower:
        if 'no_future' not in filename_lower:
            return 'kungfubot2'
    elif filename_lower == 'holosoma.txt':
        return 'holosoma'
    elif 'maskedmimic_teacher' in filename_lower or filename_lower == 'maskedmimic_teacher.txt':
        return 'maskedmimic_teacher'
    
    # Then check for variants
    if 'kungfubot1' in filename_lower or ('default' in filename_lower and 'kungfubot2' not in filename_lower):
        return 'kungfubot1'
    elif 'kungfubot2' in filename_lower or 'gmt' in filename_lower:
        return 'kungfubot2'
    elif 'holosoma' in filename_lower:
        return 'holosoma'
    elif 'maskedmimic' in filename_lower:
        return 'maskedmimic_teacher'
    return None

def assign_colors(filenames, labels):
    """Assign colors based on baseline, using fixed colors for baselines and adjusted saturation for variants"""
    colors = []
    baseline_variant_counts = {'kungfubot1': 0, 'kungfubot2': 0, 'holosoma': 0, 'maskedmimic_teacher': 0}
    
    for filename, label in zip(filenames, labels):
        baseline = detect_baseline(filename)
        
        # Check if this is a pure baseline (exact match)
        filename_lower = filename.lower()
        is_pure_baseline = (
            filename_lower == 'kungfubot1.txt' or 
            (filename_lower == 'kungfubot2_teacher_no_hands.txt' and 'no_future' not in filename_lower) or
            filename_lower == 'holosoma.txt' or
            filename_lower == 'maskedmimic_teacher.txt'
        )
        
        if is_pure_baseline and baseline:
            # Use fixed baseline color
            colors.append(BASELINE_COLORS[baseline])
        elif baseline:
            # Use baseline color with adjusted saturation (higher saturation for variants)
            base_color = BASELINE_COLORS.get(baseline, '#95A5A6')
            # Increase saturation by 20-30% for variants
            saturation_factor = 1.2 + (baseline_variant_counts.get(baseline, 0) * 0.1)
            variant_color = adjust_saturation(base_color, saturation_factor)
            colors.append(variant_color)
            baseline_variant_counts[baseline] = baseline_variant_counts.get(baseline, 0) + 1
        else:
            # Unknown baseline, use gray
            colors.append('#95A5A6')
    
    return colors

def load_models_from_folder(folder_path):
    """Load all .txt files from that folder only (category = what you put in the folder)."""
    folder = results_dir / folder_path
    if not folder.exists():
        print(f"⚠️  Folder not found: {folder_path}")
        return [], [], []
    
    txt_files = list(folder.glob('*.txt'))
    
    # Use grouped sorting
    txt_files = sort_models_grouped(txt_files)
    
    models_data = []
    model_labels = []
    filenames = []
    
    for filepath in txt_files:
        metrics = parse_metrics_file(filepath)
        if metrics:
            models_data.append(metrics)
            # Create label from filename (remove .txt, replace _ with space, title case)
            label = filepath.stem.replace('_', ' ').title()
            # Shorten common patterns
            label = label.replace('Kungfubot1', 'KungfuBot1')
            label = label.replace('Kungfubot2', 'KungfuBot2')
            label = label.replace('Maskedmimic', 'MaskedMimic')
            label = label.replace('Add Future Motion', '+ Future')
            label = label.replace('No Future', '- Future')
            label = label.replace('No Ref Pos', '- Ref Pos')
            model_labels.append(label)
            filenames.append(filepath.name)
        else:
            print(f"⚠️  Could not parse metrics from: {filepath.name}")
    
    return models_data, model_labels, filenames

def load_models(model_order):
    """Legacy function for backward compatibility"""
    models_data = []
    model_labels = []
    filenames = []
    
    for filename, label in model_order:
        filepath = results_dir / filename
        if filepath.exists():
            metrics = parse_metrics_file(filepath)
            if metrics:
                models_data.append(metrics)
                model_labels.append(label)
                filenames.append(filename)
        else:
            print(f"⚠️  File not found: {filename}")
    
    return models_data, model_labels, filenames

def create_comparison_plot(models_data, model_labels, title_suffix, filename_suffix, filenames=None):
    """Create comprehensive comparison plot"""
    
    # Extract metrics
    energy_avg = [m.get('energy_avg', 0) for m in models_data]
    slippage_avg = [m.get('slippage_avg', 0) for m in models_data]
    contact_force_avg = [m.get('contact_force_avg', 0) for m in models_data]
    contact_force_std = [m.get('contact_force_std', 0) for m in models_data]
    contact_force_imbalance = [m.get('contact_force_imbalance', 0) for m in models_data]
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
    
    # Assign colors based on baseline
    if filenames:
        model_colors = assign_colors(filenames, model_labels)
    else:
        # Fallback to default colors if filenames not provided
        model_colors = plt.cm.Set3(np.linspace(0, 1, max(n_models, 8)))[:n_models]
    
    # Display caps per metric (outlier bars truncated with wavy line + value on top)
    CAPS = {
        'energy': 25,
        'slippage': 25,
        'motion_global': 0.5,
        'motion_local': 0.5,
        'base_ang_vel': 2.0,
        'action_rate': 0.35,
        'action_smoothness': 1.0,
        'joint_acc': 80,
        'contact_imbalance': 50,
    }

    # 1. Energy
    ax1 = fig.add_subplot(gs[0, 0])
    _bar_chart_with_cap(ax1, energy_avg, model_colors, y_max=CAPS['energy'], value_fmt='.1f')
    ax1.set_ylabel('Energy (Nm·rad/s)', fontsize=10)
    ax1.set_title('1. Average Energy per Step', fontweight='bold', fontsize=11)
    ax1.set_xticks(range(n_models))
    ax1.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')

    # 2. Foot Slippage
    ax2 = fig.add_subplot(gs[0, 1])
    _bar_chart_with_cap(ax2, slippage_avg, model_colors, y_max=CAPS['slippage'], value_fmt='.1f')
    ax2.set_ylabel('Slippage (m)', fontsize=10)
    ax2.set_title('2. Foot Slippage', fontweight='bold', fontsize=11)
    ax2.set_xticks(range(n_models))
    ax2.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')

    # 3. Motion Tracking - Global Frame
    ax3 = fig.add_subplot(gs[0, 2])
    _bar_chart_with_cap(ax3, motion_tracking_global_max, model_colors, y_max=CAPS['motion_global'], value_fmt='.3f')
    ax3.set_ylabel('Max Distance (m)', fontsize=10)
    ax3.set_title('3. Motion Tracking (Global)', fontweight='bold', fontsize=11)
    ax3.set_xticks(range(n_models))
    ax3.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')

    # 4. Motion Tracking - Local Frame
    ax4 = fig.add_subplot(gs[1, 0])
    _bar_chart_with_cap(ax4, motion_tracking_local_max, model_colors, y_max=CAPS['motion_local'], value_fmt='.3f')
    ax4.set_ylabel('Max Distance (m)', fontsize=10)
    ax4.set_title('4. Motion Tracking (Local)', fontweight='bold', fontsize=11)
    ax4.set_xticks(range(n_models))
    ax4.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')

    # 5. Contact Force (Mean & Std) — cap only when value exceeds limit; wavy + value when truncated
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(n_models)
    width = 0.35
    cf_cap = 300
    cf_max = max(max(contact_force_avg), max(contact_force_std))
    use_cap = cf_max > cf_cap
    display_max = cf_cap if use_cap else cf_max * 1.15
    margin = 0.02 * display_max
    avg_display = [min(v, display_max) for v in contact_force_avg]
    std_display = [min(v, display_max) for v in contact_force_std]
    ax5.bar(x - width/2, avg_display, width, label='Mean', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1)
    ax5.bar(x + width/2, std_display, width, label='Std', color='#F39C12', alpha=0.8, edgecolor='black', linewidth=1)
    for i in range(n_models):
        av, sv = contact_force_avg[i], contact_force_std[i]
        if av > display_max:
            _draw_wavy_cap(ax5, i - width/2, display_max, bar_width=width)
            ax5.text(i - width/2, display_max + margin, f'{av:.0f}', ha='center', va='bottom', fontsize=6)
        else:
            ax5.text(i - width/2, av + margin, f'{av:.0f}', ha='center', va='bottom', fontsize=6)
        if sv > display_max:
            _draw_wavy_cap(ax5, i + width/2, display_max, bar_width=width)
            ax5.text(i + width/2, display_max + margin, f'{sv:.0f}', ha='center', va='bottom', fontsize=6)
        else:
            ax5.text(i + width/2, sv + margin, f'{sv:.0f}', ha='center', va='bottom', fontsize=6)
    ax5.set_ylim(0, display_max * 1.2)
    ax5.set_ylabel('Force (N)', fontsize=10)
    ax5.set_title('5. Contact Force (Mean & Std)', fontweight='bold', fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)

    # 6. Base Angular Velocity Error
    ax6 = fig.add_subplot(gs[1, 2])
    _bar_chart_with_cap(ax6, base_ang_vel_error, model_colors, y_max=CAPS['base_ang_vel'], value_fmt='.2f')
    ax6.set_ylabel('Error (rad/s)', fontsize=10)
    ax6.set_title('6. Base Ang Vel Error', fontweight='bold', fontsize=11)
    ax6.set_xticks(range(n_models))
    ax6.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')

    # 7. Action Rate
    ax7 = fig.add_subplot(gs[2, 0])
    _bar_chart_with_cap(ax7, action_rate, model_colors, y_max=CAPS['action_rate'], value_fmt='.3f')
    ax7.set_ylabel('Action Rate', fontsize=10)
    ax7.set_title('7. Action Rate', fontweight='bold', fontsize=11)
    ax7.set_xticks(range(n_models))
    ax7.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')

    # 8. Action Smoothness
    ax8 = fig.add_subplot(gs[2, 1])
    _bar_chart_with_cap(ax8, action_smoothness, model_colors, y_max=CAPS['action_smoothness'], value_fmt='.3f')
    ax8.set_ylabel('Smoothness (jerk²)', fontsize=10)
    ax8.set_title('8. Action Smoothness', fontweight='bold', fontsize=11)
    ax8.set_xticks(range(n_models))
    ax8.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')

    # 9. Joint Acceleration
    ax9 = fig.add_subplot(gs[2, 2])
    _bar_chart_with_cap(ax9, joint_acceleration, model_colors, y_max=CAPS['joint_acc'], value_fmt='.0f')
    ax9.set_ylabel('Accel (rad/s²)', fontsize=10)
    ax9.set_title('9. Joint Acceleration', fontweight='bold', fontsize=11)
    ax9.set_xticks(range(n_models))
    ax9.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')
    
    # 10. Contact Force Imbalance (|left mean - right mean|, lower is better)
    ax_imbalance = fig.add_subplot(gs[3, 2])
    _bar_chart_with_cap(ax_imbalance, contact_force_imbalance, model_colors, y_max=CAPS['contact_imbalance'], value_fmt='.1f')
    ax_imbalance.set_ylabel('|L−R| (N)', fontsize=10)
    ax_imbalance.set_title('10. Contact Force Imbalance', fontweight='bold', fontsize=11)
    ax_imbalance.set_xticks(range(n_models))
    ax_imbalance.set_xticklabels(model_labels, fontsize=8, rotation=45, ha='right')
    
    # 11. Overall Score (Absolute Thresholds)
    ax10 = fig.add_subplot(gs[3, :2])  # Row 4, first two columns
    
    def score_absolute(value, excellent, poor):
        """Score based on absolute thresholds (lower is better).
        excellent: 100 points, poor: 0 points. Beyond poor: negative (linear extension).
        """
        if value <= 0:
            return 0
        if value <= excellent:
            return 100
        # Linear: 100 at excellent, 0 at poor, negative when value > poor
        return 100 * (poor - value) / (poor - excellent)
    
    # Weights for overall score (higher = more impact). Contact force std & base ang vel error emphasized.
    METRIC_WEIGHTS = {
        'motion_tracking_global': 1.0,
        'motion_tracking_local': 1.0,
        'slippage': 1.0,
        'action_smoothness': 1.0,
        'contact_force_std': 2.0,   # higher: contact stability matters more
        'contact_force_imbalance': 1.5,  # left-right balance
        'joint_acceleration': 1.0,
        'base_ang_vel_error': 2.0,   # higher: orientation tracking matters more
        'energy': 1.0,
        'action_rate': 1.0,
    }
    
    # Define absolute thresholds for each metric
    overall_score = []
    for i in range(n_models):
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Motion tracking (global): excellent < 0.15m, poor > 0.5m
        if motion_tracking_global_max[i] > 0:
            s = score_absolute(motion_tracking_global_max[i], 0.15, 0.5)
            w = METRIC_WEIGHTS['motion_tracking_global']
            weighted_sum += s * w
            total_weight += w
        
        # Motion tracking (local): excellent < 0.15m, poor > 0.5m
        if motion_tracking_local_max[i] > 0:
            s = score_absolute(motion_tracking_local_max[i], 0.15, 0.5)
            w = METRIC_WEIGHTS['motion_tracking_local']
            weighted_sum += s * w
            total_weight += w
        
        # Foot slippage: excellent < 5, poor > 15
        if slippage_avg[i] > 0:
            s = score_absolute(slippage_avg[i], 4, 20)
            w = METRIC_WEIGHTS['slippage']
            weighted_sum += s * w
            total_weight += w
        
        # Action smoothness: excellent < 0.0001, poor > 0.001
        if action_smoothness[i] > 0:
            s = score_absolute(action_smoothness[i], 0.005, 0.03)
            w = METRIC_WEIGHTS['action_smoothness']
            weighted_sum += s * w
            total_weight += w
        
        # Contact force std: excellent < 20, poor > 60 (weight 2.0)
        if contact_force_std[i] > 0:
            s = score_absolute(contact_force_std[i], 100, 250)
            w = METRIC_WEIGHTS['contact_force_std']
            weighted_sum += s * w
            total_weight += w
        
        # Contact force imbalance (|left−right| mean): excellent < 15 N, poor > 55 N
        if contact_force_imbalance[i] > 0:
            s = score_absolute(contact_force_imbalance[i], 15, 55)
            w = METRIC_WEIGHTS['contact_force_imbalance']
            weighted_sum += s * w
            total_weight += w
        
        # Joint acceleration: excellent < 10, poor > 40
        if joint_acceleration[i] > 0:
            s = score_absolute(joint_acceleration[i], 10, 35)
            w = METRIC_WEIGHTS['joint_acceleration']
            weighted_sum += s * w
            total_weight += w
        
        # Base ang vel error: excellent < 0.01, poor > 0.1 (weight 2.0)
        if base_ang_vel_error[i] > 0:
            s = score_absolute(base_ang_vel_error[i], 0.4, 1.0)
            w = METRIC_WEIGHTS['base_ang_vel_error']
            weighted_sum += s * w
            total_weight += w
        
        # Energy: excellent < 4, poor > 10
        if energy_avg[i] > 0:
            s = score_absolute(energy_avg[i], 2, 10)
            w = METRIC_WEIGHTS['energy']
            weighted_sum += s * w
            total_weight += w
        
        # Action rate: excellent < 0.05, poor > 0.2
        if action_rate[i] > 0:
            s = score_absolute(action_rate[i], 0.04, 0.1)
            w = METRIC_WEIGHTS['action_rate']
            weighted_sum += s * w
            total_weight += w
        
        avg_score = weighted_sum / total_weight if total_weight > 0 else 0
        overall_score.append(avg_score)
    
    overall_score = np.array(overall_score)
    
    _bar_chart_with_cap_signed(ax10, overall_score, model_colors, y_min=-350, y_max=105, value_fmt='.1f')
    ax10.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax10.set_title('11. Overall Score (Weighted: Contact Force Std & Base Ang Vel Error ×2)', 
                   fontweight='bold', fontsize=11)
    ax10.set_xticks(range(n_models))
    ax10.set_xticklabels(model_labels, fontsize=9, rotation=45, ha='right')
    
    # Save
    output_dir = Path('/home/kyungminlee/PBHC/ablation_results')
    plt.savefig(output_dir / f'metrics_comparison_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'metrics_comparison_{filename_suffix}.pdf', bbox_inches='tight')
    
    return overall_score, model_labels

# Define folder-based comparisons
comparison_folders = [
    ('baseline_comparison', 'Baseline Comparison'),
    ('future_motion', 'Future Motion Ablation'),
    ('grounded_motion', 'Grounded Motion Ablation'),
    ('reward', 'Reward Ablation'),
    ('robot_config', 'Robot Config Ablation'),
]

def is_pure_baseline(filename):
    """Check if filename is a pure baseline"""
    name = filename.lower()
    return (
        name == 'kungfubot1.txt' or
        name == 'holosoma.txt' or
        name == 'maskedmimic_teacher.txt' or
        (name == 'kungfubot2_teacher_no_hands.txt' and 'no_future' not in name)
    )

def get_baseline_group(filename):
    """Get baseline group for a filename"""
    name = filename.lower()
    if 'kungfubot1' in name or (name == 'default.txt' and 'kungfubot2' not in name):
        return 'kungfubot1'
    elif 'kungfubot2' in name or 'gmt' in name:
        return 'kungfubot2'
    elif 'holosoma' in name:
        return 'holosoma'
    elif 'maskedmimic' in name:
        return 'maskedmimic_teacher'
    return 'other'

def get_baseline_for_variant(filename):
    """Get the baseline filename that a variant belongs to"""
    name = filename.lower()
    if 'kungfubot1' in name or ('default' in name and 'kungfubot2' not in name):
        return 'kungfubot1.txt'
    elif 'kungfubot2' in name or 'gmt' in name:
        if 'teacher' in name and 'no_hands' in name and 'no_future' not in name:
            return 'kungfubot2_teacher_no_hands.txt'
        return 'kungfubot2_teacher_no_hands.txt'
    elif 'holosoma' in name:
        return 'holosoma.txt'
    elif 'maskedmimic' in name:
        return 'maskedmimic_teacher.txt'
    return None

# Order for grouping (baseline names). Variants follow their baseline.
BASELINE_SORT_ORDER = [
    'kungfubot1.txt',
    'kungfubot2_teacher_no_hands.txt',
    'kungfubot2_student_no_hands.txt',
    'holosoma.txt',
    'maskedmimic_teacher.txt',
]


def _sort_key(filepath):
    """Group by baseline family, then by filename. Never drop a file."""
    name = filepath.name.lower()
    if is_pure_baseline(filepath.name):
        group = filepath.name.lower()
    else:
        group = get_baseline_for_variant(filepath.name) or 'other'
    try:
        order = BASELINE_SORT_ORDER.index(group)
    except ValueError:
        order = 999
    return (order, name)


def sort_models_grouped(filepaths):
    """Sort models: group by baseline family, then by name. Include every file in the folder."""
    return sorted(filepaths, key=_sort_key)

# Generate comparisons for each folder
for folder_name, title in comparison_folders:
    print("\n" + "="*80)
    print(f"COMPARISON: {title}")
    print("="*80)
    
    models_data, model_labels, filenames = load_models_from_folder(folder_name)
    
    if not models_data:
        print(f"⚠️  No models found in {folder_name}")
        continue
    
    print(f"✅ Loaded {len(models_data)} models:")
    for label, filename in zip(model_labels, filenames):
        print(f"   - {label} ({filename})")
    
    # Create filename suffix from folder name
    filename_suffix = folder_name.replace('/', '_')
    
    overall_score, labels = create_comparison_plot(
        models_data, 
        model_labels, 
        title, 
        filename_suffix,
        filenames=filenames
    )
    
    print(f"\n✅ Saved: ablation_results/metrics_comparison_{filename_suffix}.png/pdf")
    rankings = sorted(zip(labels, overall_score), key=lambda x: x[1], reverse=True)
    print("\n📊 Rankings:")
    for i, (model, score) in enumerate(rankings, 1):
        print(f"   {i}. {model:30s}: {score:5.1f}/100")

print("\n" + "="*80)
print("COMPLETE! 📊")
print("="*80)

plt.show()
