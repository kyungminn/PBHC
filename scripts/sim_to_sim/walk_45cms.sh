export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/nas4_user/kyungminlee/anaconda3/envs/pbhc/lib/
python humanoidverse/urci.py \
    +checkpoint='/home/kyungminlee/PBHC/logs/MotionTracking/20260129_062702-g1_walk_45cms-motion_tracking-g1_23dof_lock_wrist/exported/model_20000.onnx' \
    +opt=record +simulator=mujoco \
    +deploy.render=False \
    +save_embodiment_npy=True \
    +num_embodiment_steps=1000