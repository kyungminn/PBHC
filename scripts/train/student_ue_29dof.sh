cd ..
cd ..

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/kyungminlee/anaconda3/envs/pbhc/lib/

python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=2048 \
    +obs=motion_tracking/obs_ppo_student \
    +robot=g1/g1_29dof_general \
    +domain_rand=main \
    +rewards=motion_tracking/general_main \
    experiment_name=unreal_engine_motion-student-resample_100-29dof_4090gpu \
    robot.motion.motion_file="/home/kyungminlee/work/PBHC/data/g1_rig_Skeleton_Sequence_converted_processed_g1_29dof_rev_1_0.pkl" \
    algo.config.dagger_only=True \
    algo.config.teacher_model_path="/home/kyungminlee/work/PBHC/logs/MotionTracking/20260126_210808-unreal_engine_motion-teacher-resample_100-29dof_4090gpu-motion_tracking-g1_29dof_rev_1_0/model_30000.pt" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='unreal_engine_motion' \
    env.config.resample_time_interval_s=100

