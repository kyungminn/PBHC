export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=motion_tracking_priv_future +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main_priv_with_future \
    +robot=g1/g1_23dof_general \
    +domain_rand=main \
    +rewards=motion_tracking/general_main \
    experiment_name=g1_walk_45cms_with_gmt_reward_with_gmt_robot_config_priv10_future \
    robot.motion.motion_file="motion_data/g1_walk_45cms_processed_g1_23dof_lock_wrist.pkl" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='unreal_engine_motion' \
    obs.history_length=10 \
    obs.future_num_steps=20 \
    algo.config.module_dict.actor.history_encoder.tsteps=10 \
    algo.config.module_dict.actor.motion_encoder.tsteps=20 \
    algo.config.module_dict.critic.input_dim=[critic_obs,priv_obs,'"${algo.config.motion_latent_dim}"'] \
    algo.config.priv_reg_coef_schedual=[0,0.1,2000,3000] \
    algo.config.dagger_update_freq=20
