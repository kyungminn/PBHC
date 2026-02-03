export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/
# Ablation: B1 + teacher-style key body observations (without priv encoder)
# - Phase observation removed
# - Add: local_key_body_pos, local_key_body_rot, anchor_ref_pos, anchor_ref_rot, next_step_ref_motion
# - Keep everything else same: lock_wrist robot, main reward, no priv encoder, no future motion
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main_with_keybody_obs \
    +robot=g1/g1_23dof_lock_wrist \
    +domain_rand=main \
    +rewards=motion_tracking/main \
    experiment_name=g1_walk_45cms_23dof_ablation_keybody_obs \
    robot.motion.motion_file="motion_data/g1_walk_45cms_processed_g1_23dof_lock_wrist.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='unreal_engine_motion_ablation' \
    rewards.reward_scales.teleop_contact_mask=0 
