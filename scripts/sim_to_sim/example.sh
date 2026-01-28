export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/kyungminlee/anaconda3/envs/pbhc/lib/

python humanoidverse/urci.py \
    +opt=record +simulator=mujoco \
    +checkpoint=example/pretrained_horse_stance_punch/exported/model_33000.onnx
