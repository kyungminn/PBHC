export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/kyungminlee/anaconda3/envs/pbhc/lib/

# Override motion file to use specific motion data (g1_ue_walk - 23 DOF)
# Note: Playback speed is set to 2x in mujoco.py (viewer._run_speed = 2.0)
python humanoidverse/urci.py \
    +opt=record +simulator=mujoco \
    +checkpoint=/home/kyungminlee/work/PBHC/logs/MotionTracking/phuma_student/exported/model_15000.onnx \
    ++robot.asset.xml_file=g1/g1_23dof_lock_wrist_rev_2.xml \
    ++robot.motion.motion_file=/home/kyungminlee/work/PBHC/g1_ue_walk_23dof.pkl 