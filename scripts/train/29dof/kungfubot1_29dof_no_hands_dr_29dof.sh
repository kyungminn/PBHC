export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/kyungminlee/anaconda3/envs/pbhc/lib/
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main_no_hands \
    +robot=g1/g1_29dof_single_motion_no_hands \
    +domain_rand=main_29dof \
    +rewards=motion_tracking/main \
    experiment_name=kungfubot1_walk_45cms_29dof_no_hands_dr_29dof \
    robot.motion.motion_file="motion_data/g1_walk_45_pbhc.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='ablation' \
    rewards.reward_scales.teleop_contact_mask=0  \
    obs.obs_dims.10.dr_link_mass=28