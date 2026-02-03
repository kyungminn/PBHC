#!/bin/bash
# Trajectory Metrics Evaluation Script
# 
# This script runs evaluation and collects the following metrics:
# 1. Energy (per joint and average)
# 2. Foot slippage (left, right, average)
# 3. Foot contact force (left, right, average)
# 4. Action rate (per joint and average)
# 5. Motion tracking (per joint global translation distance - min, max, average)
#
# Usage:
#   ./scripts/eval/eval_trajectory_metrics.sh <checkpoint_path> [num_eval_steps]
#
# Example:
#   ./scripts/eval/eval_trajectory_metrics.sh /path/to/model_20000.pt 600

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/

# Verify Python and environment
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Set HYDRA_FULL_ERROR for detailed error messages
export HYDRA_FULL_ERROR=1

# Get checkpoint path from argument or use default
CHECKPOINT=${1:-"/home/kyungminlee/PBHC/logs/MotionTracking/20260130_064808-g1_walk_45cms_23dof_orig_add_future-motion_tracking-g1_23dof_lock_wrist/model_20000.pt"}
NUM_EVAL_STEPS=${2:-600}

echo "=========================================="
echo "Running Trajectory Metrics Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Num eval steps: $NUM_EVAL_STEPS"
echo "=========================================="

python humanoidverse/eval_metrics.py \
    +device=cuda:0 \
    +headless=True \
    +num_eval_steps=$NUM_EVAL_STEPS \
    +checkpoint=$CHECKPOINT

echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to checkpoint directory under 'metrics/'"
echo "=========================================="
