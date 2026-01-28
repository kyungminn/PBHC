cd ..
cd ..

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/kyungminlee/anaconda3/envs/pbhc/lib/

python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=5 \
    +obs=motion_tracking/obs_ppo_teacher \
    +robot=g1/g1_23dof_general \
    +domain_rand=main \
    +rewards=motion_tracking/general_main \
    experiment_name=debug-teacher \
    robot.motion.motion_file="/home/kyungminlee/PBHC/example/motion_data" \
    seed=1 \
    +device=cuda:0 \
    +opt=wandb \
    wandb.wandb_group='test' \

