export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/
# Ablation 2: Add future motion targets
# - Actor: student-style future_motion_targets (without local_ref_key_body_pos) via motion_encoder
# - Critic: has access to future_motion_local_ref_key_body_pos through priv_obs
# NOTE: Must use general_tracking exp because future motion observations are only computed in that environment
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main_with_future \
    +robot=g1/g1_23dof_lock_wrist \
    +domain_rand=main \
    +rewards=motion_tracking/main \
    experiment_name=g1_walk_45cms_23dof_ablation_future_no_ref_pos \
    robot.motion.motion_file="motion_data/g1_walk_45cms_processed_g1_23dof_lock_wrist.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='unreal_engine_motion_ablation' \
    rewards.reward_scales.teleop_contact_mask=0 
