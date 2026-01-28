cd ..
cd ..

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/kyungminlee/anaconda3/envs/pbhc/lib/

# Verify Python and environment
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Checking if AnalysisPlotMotionTracking can be imported..."
python -c "from humanoidverse.agents.callbacks.analysis_plot_motion_tracking import AnalysisPlotMotionTracking; print('Import successful')" || echo "WARNING: Import failed, but continuing..."

# Set HYDRA_FULL_ERROR for detailed error messages
export HYDRA_FULL_ERROR=1

python humanoidverse/eval_agent.py \
    +device=cuda:0 \
    +env.config.enforce_randomize_motion_start_eval=False \
    +checkpoint='/home/kyungminlee/PBHC/logs/MotionTracking/20251117_104724-phuma-student-resample_1000-gamma_0.98-teacher_50000-motion_tracking-g1_23dof_lock_wrist/model_15000.pt' \




