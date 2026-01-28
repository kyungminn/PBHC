export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/kyungminlee/anaconda3/envs/pbhc/lib/

# Note: checkpoint config already contains all 29 DOF settings
# Only override xml_file for MuJoCo (different from IsaacGym)
python humanoidverse/urci.py \
    +opt=record +simulator=mujoco \
    +checkpoint=/home/kyungminlee/work/PBHC/logs/MotionTracking/20260127_101136-unreal_engine_motion-student-resample_100-29dof_4090gpu-motion_tracking-g1_29dof_rev_1_0/exported/model_10000.onnx \
    ++robot.asset.xml_file=g1/g1_29dof_rev_1_0.xml \
    ++robot.motion.motion_file=/home/kyungminlee/work/PBHC/g1_rig_Skeleton_Sequence_converted_processed_g1_29dof_rev_1_0.pkl