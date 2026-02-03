import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from utils.config_utils import *  # noqa: E402, F403

# add argparse arguments
from utils.devtool import pdb_decorator
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger

import threading
# from pynput import keyboard

def on_press(key, env):
    try:
        if key.char == 'n':
            env.next_task()
            logger.info("Moved to the next task.")
        # Force Control
       # Force Control
        if hasattr(key, 'char'):
            if key.char == '1':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] += 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '2':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] -= 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '3':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] += 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
            elif key.char == '4':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] -= 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
    except AttributeError:
        pass

def listen_for_keypress(env):
    return
    with keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
        listener.join()



# from humanoidverse.envs.base_task.base_task import BaseTask
# from humanoidverse.envs.base_task.omnih2o_cfg import OmniH2OCfg
@hydra.main(config_path="config", config_name="base_eval")
# @pdb_decorator
def main(override_config: OmegaConf):
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config
            
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    if simulator_type == 'IsaacGym':
        import isaacgym
        
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    import torch
    from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_encoder_as_onnx

    pre_process_config(config)

    # use config.device if specified, otherwise use cuda if available
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    # print(f"config.num_envs: {config.num_envs}"); breakpoint()
    ckpt_num = config.checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
    config.num_envs = 1
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    config.env.config.ckpt_dir = str(checkpoint.parent) # commented out for now, might need it back to save motion
    
    # Configure motion saving before environment instantiation
    save_inference_motion = config.get("save_inference_motion", False)
    if save_inference_motion:
        num_eval_steps = config.get("num_eval_steps", 300)
        config.env.config.save_motion = True
        config.env.config.save_total_steps = num_eval_steps
        config.env.config.save_note = "inference"
        import time
        config.env.config.eval_timestamp = time.strftime("%Y%m%d_%H%M%S")
        logger.info(f"Motion saving configured: {num_eval_steps} steps")
    
    env = instantiate(config.env, device=device)

    # Start a thread to listen for key press
    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()

    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    EXPORT_POLICY = False
    EXPORT_ONNX = os.environ.get("EXPORT_ONNX", "True").lower() in ("true", "1")

    checkpoint_path = str(checkpoint)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # from checkpoint path

    ROBOVERSE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exported_policy_path = os.path.join(ROBOVERSE_ROOT_DIR, checkpoint_dir, 'exported')
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split('/')[-1]
    exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')

    if EXPORT_POLICY:
        export_policy_as_jit(algo.alg.actor_critic, exported_policy_path, exported_policy_name)
        logger.info('Exported policy as jit script to: ', os.path.join(exported_policy_path, exported_policy_name))
    if EXPORT_ONNX:
        example_obs_dict = algo.get_example_obs()
        if "ppo_mimic" in algo.__class__.__module__:
            export_policy_and_encoder_as_onnx(
                algo.inference_model,
                exported_policy_path,
                exported_onnx_name,
                example_obs_dict,
            )
        else:
            export_policy_as_onnx(algo.inference_model, exported_policy_path, exported_onnx_name, example_obs_dict)

        logger.info(f'Exported policy as onnx to: {os.path.join(exported_policy_path, exported_onnx_name)}')

    # Check if motion saving is enabled and run evaluation
    if save_inference_motion:
        # Disable file writing, we'll handle saving manually
        env._write_to_file = False
        
        logger.info(f"Running evaluation for {num_eval_steps} steps to collect motion data...")
        
        # Run evaluation for specified number of steps
        algo.evaluate_policy_steps(num_eval_steps + 5)
        
        # Save the motion data to npy file
        if hasattr(env, 'saved_motion_dict'):
            import numpy as np
            inference_motion_dir = checkpoint.parent / "inference_motion"
            inference_motion_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to format compatible with visualization script
            # The script expects: root_trans, root_ori, dof_pos, fps
            motion_data_to_save = {
                'root_trans': env.saved_motion_dict['root_trans_offset'][0],  # Take first env
                'root_ori': env.saved_motion_dict['root_rot'][0],  # xyzw format
                'dof_pos': env.saved_motion_dict['dof'][0],
                'fps': int(1 / env.dt)
            }
            
            save_path = inference_motion_dir / f"inference_motion_ckpt_{ckpt_num}.npy"
            np.save(save_path, motion_data_to_save)
            logger.info(f"Saved inference motion to {save_path}")
            logger.info(f"  - root_trans shape: {motion_data_to_save['root_trans'].shape}")
            logger.info(f"  - root_ori shape: {motion_data_to_save['root_ori'].shape}")
            logger.info(f"  - dof_pos shape: {motion_data_to_save['dof_pos'].shape}")
            logger.info(f"  - fps: {motion_data_to_save['fps']}")
        else:
            logger.warning("saved_motion_dict not found in environment")
    else:
        # Normal evaluation mode
        algo.evaluate_policy()


if __name__ == "__main__":
    main()
