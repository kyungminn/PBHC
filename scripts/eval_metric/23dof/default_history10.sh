export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/

# Verify Python and environment
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Checking if AnalysisPlotMotionTracking can be imported..."
python -c "from humanoidverse.agents.callbacks.analysis_plot_motion_tracking import AnalysisPlotMotionTracking; print('Import successful')" || echo "WARNING: Import failed, but continuing..."

# Set HYDRA_FULL_ERROR for detailed error messages
export HYDRA_FULL_ERROR=1

# Optional: Set custom save path by uncommenting and modifying the line below
# SAVE_PATH="your/custom/path/results.txt"

python humanoidverse/eval_metrics.py \
    +device=cuda:0 \
    +headless=True \
    +num_eval_steps=600 \
    +env.config.enforce_randomize_motion_start_eval=False \
    +checkpoint=/home/kyungminlee/PBHC/logs/MotionTracking/20260130_054026-g1_walk_45cms_23dof_ablation_history10-motion_tracking-g1_23dof_lock_wrist/model_20000.pt \
    ++metrics_save_path=./ablation_results/default_history10.txt
