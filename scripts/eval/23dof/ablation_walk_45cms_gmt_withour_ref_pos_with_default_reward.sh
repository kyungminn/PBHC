export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/

# Verify Python and environment
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Checking if AnalysisPlotMotionTracking can be imported..."
python -c "from humanoidverse.agents.callbacks.analysis_plot_motion_tracking import AnalysisPlotMotionTracking; print('Import successful')" || echo "WARNING: Import failed, but continuing..."

# Set HYDRA_FULL_ERROR for detailed error messages
export HYDRA_FULL_ERROR=1

python humanoidverse/eval_agent.py \
    +device=cuda:0 \
    +headless=True \
    +save_inference_motion=True \
    +num_eval_steps=600 \
    +env.config.enforce_randomize_motion_start_eval=False \
    +checkpoint=/home/kyungminlee/PBHC/logs/MotionTracking/20260130_055200-g1_walk_45cms_23dof_ablation_future_no_ref_pos-motion_tracking-g1_23dof_lock_wrist/model_23000.pt



