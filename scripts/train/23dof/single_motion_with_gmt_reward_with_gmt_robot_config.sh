export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main \
    +robot=g1/g1_23dof_general \
    +domain_rand=main \
    +rewards=motion_tracking/general_main \
    experiment_name=g1_walk_45cms_with_gmt_reward_with_gmt_robot_config \
    robot.motion.motion_file="motion_data/g1_walk_45cms_processed_g1_23dof_lock_wrist.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='unreal_engine_motion' \