export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=motion_tracking_priv +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main_priv \
    +robot=g1/g1_23dof_lock_wrist \
    +domain_rand=main \
    +rewards=motion_tracking/main \
    experiment_name=g1_walk_45cms_23dof_use_history_length10_estimate_priv \
    robot.motion.motion_file="motion_data/g1_walk_45cms_processed_g1_23dof_lock_wrist.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='unreal_engine_motion' \
    rewards.reward_scales.teleop_contact_mask=0 \
    obs.history_length=10 \
    algo.config.module_dict.actor.history_encoder.tsteps=10 \
    algo.config.priv_reg_coef_schedual=[0,0.1,2000,3000] \
    algo.config.dagger_update_freq=20

