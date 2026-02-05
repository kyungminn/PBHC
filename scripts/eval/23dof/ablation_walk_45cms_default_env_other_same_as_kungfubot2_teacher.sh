export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/

# Verify Python and environment
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Checking if AnalysisPlotMotionTracking can be imported..."
python -c "from humanoidverse.agents.callbacks.analysis_plot_motion_tracking import AnalysisPlotMotionTracking; print('Import successful')" || echo "WARNING: Import failed, but continuing..."

# Set HYDRA_FULL_ERROR for detailed error messages
export HYDRA_FULL_ERROR=1

EXPORT_ONNX=False python humanoidverse/eval_agent.py \
    +device=cuda:0 \
    +headless=True \
    +save_inference_motion=True \
    +num_eval_steps=600 \
    +checkpoint=/home/kyungminlee/PBHC/logs/MotionTracking/20260201_123010-g1_walk_45cms_with_gmt_reward_with_gmt_robot_config_priv10_future_add_ref_pos-motion_tracking-g1_23dof_lock_wrist/model_20000.pt



