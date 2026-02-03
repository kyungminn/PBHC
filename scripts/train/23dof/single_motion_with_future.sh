export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/
# Motion tracking with future_motion_targets using motion_encoder
# - Uses mh_ppo with motion_encoder (mh_ppo_motion algo)
# - motion_encoder output (h_t) is concatenated to both actor_obs and critic_obs
# - Actor obs: deployable (no privileged info)
# - Critic obs: includes privileged info for better value estimation
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=motion_tracking_future +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main_with_future_ \
    +robot=g1/g1_23dof_lock_wrist \
    +domain_rand=main \
    +rewards=motion_tracking/main \
    experiment_name=g1_walk_45cms_23dof_orig_add_future \
    robot.motion.motion_file="motion_data/g1_walk_45cms_processed_g1_23dof_lock_wrist.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='unreal_engine_motion_ablation' \
    rewards.reward_scales.teleop_contact_mask=0 
