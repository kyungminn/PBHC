export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/
# Motion tracking with future_motion_targets using motion_encoder
# - Uses mh_ppo with motion_encoder (mh_ppo_motion algo)
# - motion_encoder output (h_t) is concatenated to both actor_obs and critic_obs
# - Actor obs: deployable (no privileged info)
# - Critic obs: includes privileged info for better value estimation
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=motion_tracking_future +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main_with_future_add_trans \
    +robot=g1/g1_23dof_lock_wrist \
    +domain_rand=main \
    +rewards=motion_tracking/main \
    experiment_name=kungfubot1_with_future_motion_fall_and_getup \
    robot.motion.motion_file="motion_data/fall_and_getup_23dof.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='unreal_engine_motion_ablation' \
    rewards.reward_scales.teleop_contact_mask=0 \
    algo.config.module_dict.actor.motion_encoder.tsteps=20 \
    algo.config.module_dict.actor.motion_encoder.input_dim=[36] \
    algo.config.module_dict.actor.motion_encoder.hidden_dim=72 
