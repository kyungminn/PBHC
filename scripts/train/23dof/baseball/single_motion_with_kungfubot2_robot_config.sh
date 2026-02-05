export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main \
    +robot=g1/g1_23dof_general \
    +domain_rand=main \
    +rewards=motion_tracking/main \
    experiment_name=kungfubot1_with_kungfubot2_robot_config_baseball_batter \
    robot.motion.motion_file="motion_data/baseball_batter_23dof.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='ablation' \
    rewards.reward_scales.teleop_contact_mask=0 