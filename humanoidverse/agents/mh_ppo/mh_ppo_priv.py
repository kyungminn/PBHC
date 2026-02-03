"""
MHPPO with Privileged Encoder and History Encoder support.

This variant adds:
- priv_encoder: Encodes privileged observations (priv_obs) into a latent
- history_encoder: Encodes proprioceptive history (prop_history) into a latent

During training:
- Actor receives: actor_obs + priv_latent
- Critic receives: critic_obs + priv_latent
- priv_reg_loss: L2 distance between priv_latent and hist_latent (trains priv_encoder to match hist_encoder)

During inference:
- Actor receives: actor_obs + hist_latent (history_encoder replaces priv_encoder)
"""

import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_priv_modules import PPOActorPriv, PPOCriticPriv
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
from typing import List, Dict, Any
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live

console = Console()


class MHPPOPriv(BaseAlgo):
    """MHPPO with Privileged Encoder and History Encoder support."""
    
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):

        self.device = device
        self.env = env
        self.config = config
        self.log_dir = log_dir
        
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Book keeping
        self.ep_infos: List[Dict[str, Any]] = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.eval_callbacks: list[RL_EvalCallback] = []
        self.episode_env_tensors = TensorAverageMeterDict()
        _ = self.env.reset_all()
        
        # Counter for priv_reg_loss scheduling
        self.counter = 0

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.env.config.robot.actions_dim

        # Logging related Config
        self.save_interval = self.config.save_interval
        self.logging_interval = self.config.get('logging_interval', 10)

        # Training related Config
        self.num_steps_per_env = self.config.num_steps_per_env
        self.load_optimizer = self.config.load_optimizer
        self.num_learning_iterations = self.config.num_learning_iterations
        self.init_at_random_ep_len = self.config.init_at_random_ep_len

        # Algorithm related Config
        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule
        self.actor_learning_rate = self.config.actor_learning_rate
        self.critic_learning_rate = self.config.critic_learning_rate
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss
        self.cfg_l2c2 = self.config.l2c2 if 'l2c2' in self.config else None

        self.num_rew_fn = self.env.num_rew_fn
        
        # Priv reg loss scheduling: [start_coef, end_coef, start_iter, ramp_iters]
        self.priv_reg_coef_schedual = self.config.get('priv_reg_coef_schedual', [0, 0.1, 2000, 3000])
        
        # Dagger update frequency for history encoder
        self.dagger_update_freq = self.config.get('dagger_update_freq', 20)
        
        # Check if future_motion_targets is in obs (for motion_encoder support)
        self.use_motion_encoder = 'future_motion_targets' in self.env.config.obs.obs_dict
        
        # Check if priv_obs and prop_history are in obs
        self.use_priv_encoder = 'priv_obs' in self.env.config.obs.obs_dict
        self.use_history_encoder = 'prop_history' in self.env.config.obs.obs_dict

    def setup(self):
        logger.info("Setting up MHPPOPriv")
        self._setup_models_and_optimizer()
        logger.info(f"Setting up Storage")
        self._setup_storage()

    def _setup_models_and_optimizer(self):
        self.config.module_dict.critic['output_dim'][-1] = self.num_rew_fn
        
        actor_kwargs = {
            "obs_dim_dict": self.algo_obs_dim_dict,
            "module_config_dict": self.config.module_dict.actor,
            "num_actions": self.num_act,
            "init_noise_std": self.config.init_noise_std
        }
        critic_kwargs = {
            "obs_dim_dict": self.algo_obs_dim_dict,
            "module_config_dict": self.config.module_dict.critic,
        }
        
        # Use PPOActorPriv and PPOCriticPriv
        self.actor = PPOActorPriv(**actor_kwargs).to(self.device)
        self.critic = PPOCriticPriv(**critic_kwargs).to(self.device)
            
        print(self.actor)
        print(self.critic)
        
        # Separate optimizer for history encoder (trained with dagger)
        if self.use_history_encoder:
            hist_encoder_params = list(self.actor.history_encoder.parameters())
            other_actor_params = [p for p in self.actor.parameters() if not any(p is hp for hp in hist_encoder_params)]
            self.actor_optimizer = optim.Adam(other_actor_params, lr=self.actor_learning_rate)
            self.hist_encoder_optimizer = optim.Adam(hist_encoder_params, lr=self.actor_learning_rate)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
            self.hist_encoder_optimizer = None
            
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env, self.device)
        
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
            self.storage.register_key('next_'+obs_key, shape=(obs_dim,), dtype=torch.float)
        
        ## Register others
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('rewards', shape=(self.num_rew_fn,), dtype=torch.float)
        self.storage.register_key('dones', shape=(1,), dtype=torch.bool)
        self.storage.register_key('values', shape=(self.num_rew_fn,), dtype=torch.float)
        self.storage.register_key('returns', shape=(self.num_rew_fn,), dtype=torch.float)
        self.storage.register_key('advantages', shape=(1,), dtype=torch.float)
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float)
        self.storage.register_key('action_mean', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('action_sigma', shape=(self.num_act,), dtype=torch.float)

    def _eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def _train_mode(self):
        self.actor.train()
        self.critic.train()

    def load(self, ckpt_path):
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                if self.hist_encoder_optimizer is not None and "hist_encoder_optimizer_state_dict" in loaded_dict:
                    self.hist_encoder_optimizer.load_state_dict(loaded_dict["hist_encoder_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rate}")
            self.current_learning_iteration = loaded_dict["iter"]
            self.counter = loaded_dict.get("counter", 0)
            return loaded_dict["infos"]

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        save_dict = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'counter': self.counter,
            'infos': infos,
        }
        if self.hist_encoder_optimizer is not None:
            save_dict['hist_encoder_optimizer_state_dict'] = self.hist_encoder_optimizer.state_dict()
        torch.save(save_dict, path)
        
    def learn(self):
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
            
        self._train_mode()

        num_learning_iterations = self.num_learning_iterations
        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        for it in range(self.current_learning_iteration, tot_iter):
            # Determine if this iteration should update history encoder
            self.hist_encoding = it % self.dagger_update_freq == 0
            self.start_time = time.time()

            obs_dict = self._rollout_step(obs_dict)

            loss_dict = self._training_step()
            
            # Update history encoder with dagger
            if self.hist_encoding and self.use_history_encoder:
                loss_dict = self._training_step_dagger(loss_dict)

            self.stop_time = time.time()
            self.learn_time = self.stop_time - self.start_time

            # Logging
            log_dict = {
                'it': it,
                'loss_dict': loss_dict,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'num_learning_iterations': num_learning_iterations
            }
            self._post_epoch_logging(log_dict)
            if it % self.save_interval == 0:
                self.current_learning_iteration = it
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            self.ep_infos.clear()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def _get_priv_latent(self, obs_dict):
        """Get privileged latent from priv_obs."""
        if self.use_priv_encoder:
            return self.actor.priv_encoding(obs_dict["priv_obs"])
        return None
    
    def _get_hist_latent(self, obs_dict):
        """Get history latent from prop_history."""
        if self.use_history_encoder:
            return self.actor.history_encoding(obs_dict["prop_history"])
        return None

    def _actor_rollout_step(self, obs_dict, policy_state_dict):
        # During rollout, use priv_latent (not hist_latent)
        priv_latent = self._get_priv_latent(obs_dict)
        motion_embedding = None
        if self.use_motion_encoder:
            motion_embedding = self.actor.motion_encoding(obs_dict["future_motion_targets"])
        
        actions = self.actor.act(obs_dict["actor_obs"], latent=priv_latent, motion_embedding=motion_embedding)
        policy_state_dict["actions"] = actions
        
        action_mean = self.actor.action_mean.detach()
        action_sigma = self.actor.action_std.detach()
        actions_log_prob = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob

        return policy_state_dict

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                
                # Get priv_latent for critic
                priv_latent = self._get_priv_latent(obs_dict)
                motion_embedding = None
                if self.use_motion_encoder:
                    motion_embedding = self.actor.motion_encoding(obs_dict["future_motion_targets"])
                # Critic uses raw priv_obs instead of encoded priv_latent (like ppo_mimic)
                values = self.critic.evaluate(obs_dict["critic_obs"], priv_latent=obs_dict["priv_obs"], motion_embedding=motion_embedding).detach()
                policy_state_dict["values"] = values

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                    
                actions = policy_state_dict["actions"]
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                    self.storage.update_key('next_'+obs_key, obs_dict[obs_key])
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().reshape(self.env.num_envs, self.env.num_rew_fn)

                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards.view(self.env.num_envs, self.env.num_rew_fn).sum(dim=-1)
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # Compute returns
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(
                    values=self.storage.query_key('values'), 
                    dones=self.storage.query_key('dones'), 
                    rewards=self.storage.query_key('rewards')
                )
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        self.actor.reset(dones)
        self.critic.reset(dones)

    def _compute_returns(self, last_obs_dict, policy_state_dict):
        """Compute returns and advantages using GAE."""
        priv_latent = self._get_priv_latent(last_obs_dict)
        motion_embedding = None
        if self.use_motion_encoder:
            motion_embedding = self.actor.motion_encoding(last_obs_dict["future_motion_targets"])
        # Critic uses raw priv_obs instead of encoded priv_latent
        last_values = self.critic.evaluate(last_obs_dict["critic_obs"], priv_latent=last_obs_dict["priv_obs"], motion_embedding=motion_embedding).detach()
        
        values = policy_state_dict['values']
        dones = policy_state_dict['dones']
        rewards = policy_state_dict['rewards']
        
        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        
        returns = torch.zeros_like(values)
        
        num_steps = returns.shape[0]
        advantage = 0
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize advantages
        tot_advantages = returns - values
        aggr_tot_advantages = tot_advantages.sum(dim=-1)
        advantages = (aggr_tot_advantages - aggr_tot_advantages.mean()) / (aggr_tot_advantages.std() + 1e-8)
        return returns, advantages.unsqueeze(-1)
    
    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for policy_state_dict in generator:
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        self.storage.clear()
        self.update_counter()
        return loss_dict
    
    def _training_step_dagger(self, loss_dict):
        """Update history encoder to match priv_encoder output."""
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        hist_latent_loss_total = 0
        num_updates = 0
        
        for policy_state_dict in generator:
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            
            # Get priv_latent (frozen)
            with torch.inference_mode():
                priv_latent_batch = self.actor.priv_encoding(policy_state_dict["priv_obs"])
            
            # Get hist_latent (trainable)
            hist_latent_batch = self.actor.history_encoding(policy_state_dict["prop_history"])
            
            # L2 loss
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            
            hist_latent_loss_total += hist_latent_loss.item()
            num_updates += 1
        
        loss_dict["hist_latent_loss"] = hist_latent_loss_total / num_updates
        return loss_dict
    
    def _init_loss_dict_at_training_step(self):
        loss_dict = {
            'Value': 0,
            'Surrogate': 0,
            'Entropy': 0,
            'priv_reg_loss': 0,
            'L2C2_Value': 0,
            'L2C2_Policy': 0
        }
        return loss_dict
    
    def _update_algo_step(self, policy_state_dict, loss_dict):
        loss_dict = self._update_ppo(policy_state_dict, loss_dict)
        return loss_dict

    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        # Get latents
        priv_latent = self.actor.priv_encoding(policy_state_dict["priv_obs"]) if self.use_priv_encoder else None
        motion_embedding = None
        if self.use_motion_encoder:
            motion_embedding = self.actor.motion_encoding(policy_state_dict["future_motion_targets"])

        # Forward pass for actor
        self.actor.act(policy_state_dict["actor_obs"], latent=priv_latent, motion_embedding=motion_embedding)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        
        # Forward pass for critic (use raw priv_obs instead of encoded priv_latent)
        priv_obs_critic = policy_state_dict["priv_obs"]
        motion_embedding_critic = motion_embedding.detach() if motion_embedding is not None else None
        value_batch = self.critic.evaluate(policy_state_dict["critic_obs"], priv_latent=priv_obs_critic, motion_embedding=motion_embedding_critic)
        
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        # Priv reg loss: train priv_encoder to produce output similar to hist_encoder
        priv_reg_loss = torch.tensor(0.0, device=self.device)
        if self.use_priv_encoder and self.use_history_encoder:
            with torch.inference_mode():
                hist_latent_batch = self.actor.history_encoding(policy_state_dict["prop_history"])
            priv_reg_loss = (priv_latent - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            
            # Scheduled coefficient
            priv_reg_stage = min(
                max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3],
                1,
            )
            priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]
        else:
            priv_reg_coef = 0

        # KL divergence for learning rate adaptation
        if self.desired_kl is not None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                    (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, 
                    axis=-1
                )
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).sum(dim=-1).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).sum(dim=-1).mean()

        entropy_loss = entropy_batch.mean()
        
        # L2C2 smooth (optional)
        l2c2_value_loss = torch.tensor(0.0, device=self.device)
        l2c2_policy_loss = torch.tensor(0.0, device=self.device)
        if self.cfg_l2c2 is not None and self.cfg_l2c2.enable:
            lam_value = self.cfg_l2c2.lambda_value
            lam_policy = self.cfg_l2c2.lambda_policy
            actor_obs, next_actor_obs = policy_state_dict['actor_obs'], policy_state_dict['next_actor_obs']
            critic_obs, next_critic_obs = policy_state_dict['critic_obs'], policy_state_dict['next_critic_obs']
            priv_obs, next_priv_obs = policy_state_dict['priv_obs'], policy_state_dict['next_priv_obs']
            
            u = torch.rand(*actor_obs.shape[:-1], 1, device=self.device) * 2 - 1
            u_actor_obs = actor_obs + u * (next_actor_obs - actor_obs)
            u_critic_obs = critic_obs + u * (next_critic_obs - critic_obs)
            u_priv_obs = priv_obs + u * (next_priv_obs - priv_obs)
            
            u_mu = self.actor.act_inference(u_actor_obs, latent=priv_latent.detach() if priv_latent is not None else None, motion_embedding=motion_embedding.detach() if motion_embedding is not None else None)
            u_value = self.critic.evaluate(u_critic_obs, priv_latent=u_priv_obs, motion_embedding=motion_embedding_critic)
            
            l2c2_value_loss = lam_value * (value_batch - u_value).pow(2).mean()
            l2c2_policy_loss = lam_policy * (actions_batch - u_mu).pow(2).mean()
        
        # Total losses
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss + l2c2_policy_loss + priv_reg_coef * priv_reg_loss
        critic_loss = self.value_loss_coef * value_loss + l2c2_value_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict['priv_reg_loss'] += priv_reg_loss.item()
        loss_dict['L2C2_Value'] += l2c2_value_loss.item()
        loss_dict['L2C2_Policy'] += l2c2_policy_loss.item()
        return loss_dict
    
    def update_counter(self):
        self.counter += 1

    def set_learning_rate(self, actor_learning_rate, critic_learning_rate):
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

    @property
    def inference_model(self):
        return {
            "actor": self.actor,
            "critic": self.critic
        }

    def _post_epoch_logging(self, log_dict, width=80, pad=40):
        # Update total timesteps and total time
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']
        
        if log_dict['it'] % self.logging_interval != 0:
            return

        def generate_computation_log():
            mean_std = self.actor.std.mean()
            fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
            str_header = f" \033[1m Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "
            
            return (f"""{str_header.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std:>10.4f}\n""")

        def generate_reward_length_log():
            reward_length_string = ""
            if len(log_dict['rewbuffer']) > 0:
                reward_length_string += (f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):>10.4f}\n"""
                                         f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):>10.4f}\n""")
                self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
                self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            return reward_length_string

        def generate_env_log():
            env_log_string = ""
            env_log_dict = self.episode_env_tensors.mean_and_clear()
            env_log_dict = {f"{k}": v for k, v in env_log_dict.items()}
            
            for k, v in env_log_dict.items():
                entry = f"{f'{k}:':>{pad}} {v:>10.4f}"
                env_log_string += f"{entry}\n"
                self.writer.add_scalar('Env/'+k, v, log_dict['it'])
                
            for loss_key, loss_value in log_dict['loss_dict'].items():
                self.writer.add_scalar(f'Learn/{loss_key}', loss_value, log_dict['it'])
            self.writer.add_scalar('Learn/actor_learning_rate', self.actor_learning_rate, log_dict['it'])
            self.writer.add_scalar('Learn/critic_learning_rate', self.critic_learning_rate, log_dict['it'])
            self.writer.add_scalar('Learn/mean_noise_std', self.actor.std.mean().item(), log_dict['it'])
            
            return env_log_string

        def generate_episode_log():
            ep_string = f"{'-' * width}\n"
            
            if log_dict['ep_infos']:
                mean_values = {key: 0.0 for key in log_dict['ep_infos'][0].keys()}
                total_episodes = 0

                for ep_info in log_dict['ep_infos']:
                    for key in mean_values.keys():
                        if key == 'end_epis_length':
                            mean_values[key] += ep_info[key].sum().item()
                            total_episodes += ep_info[key].numel()
                        else:
                            mean_values[key] += (ep_info[key] / ep_info['end_epis_length'] * self.env.max_episode_length).sum().item()

                rew_total = sum(v for k, v in mean_values.items() if k.startswith('rew_'))
                mean_values['rew_total'] = rew_total
                
                for key in mean_values.keys():
                    mean_values[key] /= total_episodes
                    self.writer.add_scalar('Env/' + key, mean_values[key], log_dict['it'])
                        
                for key, value in mean_values.items():
                    if key == 'end_epis_length': 
                        continue
                    ep_string += f"""{f'{key}:':>{pad}} {value:>10.4f} \n"""
            ep_string += f"Note: reward computed per step\n"

            return ep_string

        def generate_total_time_log():
            fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
            eta = self.tot_time / (log_dict['it'] + 1) * (log_dict['num_learning_iterations'] - log_dict['it'])
            
            self.writer.add_scalar('Perf/total_fps', fps, log_dict['it'])
            self.writer.add_scalar('Perf/collection_time', log_dict['collection_time'], log_dict['it'])
            self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
            self.writer.add_scalar('Perf/iter_time', iteration_time, log_dict['it'])
            self.writer.add_scalar('Perf/total_time', self.tot_time, log_dict['it'])
        
            return (f"""{'-' * width}\n"""
                    f"""{'Total timesteps:':>{pad}} {self.tot_timesteps:.0f}\n"""
                    f"""{'Collection time:':>{pad}} {log_dict['collection_time']:>10.4f}s\n"""
                    f"""{'Learning time:':>{pad}} {log_dict['learn_time']:>10.4f}s\n"""
                    f"""{'Iteration time:':>{pad}} {iteration_time:>10.4f}s\n"""
                    f"""{'Total time:':>{pad}} {self.tot_time:>10.4f}s\n"""
                    f"""{'ETA:':>{pad}} {eta:>10.4f}s\n"""
                    f"""{'Time Now:':>{pad}} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n""")

        log_string = (generate_computation_log() +
                      generate_reward_length_log() +
                      generate_env_log() +
                      generate_episode_log() +
                      generate_total_time_log() +
                      f"Logging Directory: {self.log_dir}")

        with Live(Panel(log_string, title="Training Log"), refresh_per_second=4, console=console):
            pass

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @torch.no_grad()
    def evaluate_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1
        self._post_evaluate_policy()

    @torch.no_grad()
    def evaluate_policy_steps(self, Nsteps: int):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        while step < Nsteps:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1
        self._post_evaluate_policy()

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        obs_dict = actor_state["obs"]
        
        # During inference, use hist_latent instead of priv_latent
        hist_latent = self._get_hist_latent(obs_dict) if self.use_history_encoder else None
        motion_embedding = None
        if self.use_motion_encoder:
            motion_embedding = self.actor.motion_encoding(obs_dict['future_motion_targets'])
        
        actions = self.actor.act_inference(obs_dict['actor_obs'], latent=hist_latent, motion_embedding=motion_embedding)
        actor_state.update({"actions": actions})
        
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def _get_inference_policy(self, device=None):
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        
        def inference_policy(actor_obs, prop_history=None, future_motion_targets=None):
            """Inference policy that uses history encoder."""
            hist_latent = None
            if self.use_history_encoder and prop_history is not None:
                hist_latent = self.actor.history_encoding(prop_history)
            motion_embedding = None
            if self.use_motion_encoder and future_motion_targets is not None:
                motion_embedding = self.actor.motion_encoding(future_motion_targets)
            return self.actor.act_inference(actor_obs, latent=hist_latent, motion_embedding=motion_embedding)
        
        return inference_policy

