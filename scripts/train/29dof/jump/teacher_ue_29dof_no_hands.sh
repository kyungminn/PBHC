export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/kyungminlee/anaconda3/envs/pbhc/lib/

python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/obs_ppo_teacher_no_hands \
    +robot=g1/g1_29dof_general_no_vr \
    +domain_rand=main \
    +rewards=motion_tracking/general_main \
    experiment_name=kungfubot2_teacher_jump_29dof_no_hands \
    robot.motion.motion_file="motion_data/jump.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='ablation' \
    env.config.resample_time_interval_s=100 \

