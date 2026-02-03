from __future__ import annotations
from copy import deepcopy

from typing import List
import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import BaseModule
from .encoder_modules import ConvEncoder


class PPOActorPriv(nn.Module):
    """PPO Actor with privileged encoder and history encoder support.
    
    During training: uses priv_encoder to encode priv_obs
    During inference: uses history_encoder to encode prop_history
    """
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions,
                init_noise_std):
        super(PPOActorPriv, self).__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)
        
        # Check if motion_encoder is configured
        self.use_motion_encoder = hasattr(module_config_dict, 'motion_encoder') and module_config_dict.motion_encoder is not None
        
        if self.use_motion_encoder:
            # Build motion encoder
            self.motion_encoder = ConvEncoder(
                obs_dim_dict,
                module_config_dict.motion_encoder,
                module_config_dict.motion_encoder.tsteps,
            )
            self.motion_encoder_output_dim = module_config_dict.motion_encoder.output_dim

        # Build history encoder
        self.use_history_encoder = hasattr(module_config_dict, 'history_encoder') and module_config_dict.history_encoder is not None
        if self.use_history_encoder:
            new_obs_dim_dict = obs_dim_dict.copy()
            new_obs_dim_dict["prop_history"] //= module_config_dict.history_encoder.tsteps
            self.history_encoder = ConvEncoder(
                new_obs_dim_dict,
                module_config_dict.history_encoder,
                module_config_dict.history_encoder.tsteps,
            )
            self.history_encoder_output_dim = module_config_dict.history_encoder.output_dim
        else:
            self.history_encoder = None
            self.history_encoder_output_dim = 0

        # Build priv encoder
        self.use_priv_encoder = hasattr(module_config_dict, 'priv_encoder') and module_config_dict.priv_encoder is not None
        if self.use_priv_encoder:
            self.priv_encoder = BaseModule(obs_dim_dict, module_config_dict.priv_encoder)
            # output_dim can be a list or int
            output_dim = module_config_dict.priv_encoder.output_dim
            self.priv_encoder_output_dim = sum(output_dim) if isinstance(output_dim, (list, tuple)) else output_dim
        else:
            self.priv_encoder = None
            self.priv_encoder_output_dim = 0

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module
    
    def motion_encoding(self, motion_obs):
        """Encode future motion targets"""
        return self.motion_encoder(motion_obs)
    
    def history_encoding(self, history_obs):
        """Encode proprioceptive history"""
        return self.history_encoder(history_obs)
    
    def priv_encoding(self, priv_obs):
        """Encode privileged observations"""
        return self.priv_encoder(priv_obs)
    
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs, latent=None, motion_embedding=None):
        """Update action distribution with optional latent and motion embeddings."""
        parts = [actor_obs]
        if motion_embedding is not None:
            parts.append(motion_embedding)
        if latent is not None:
            parts.append(latent)
        actor_input = torch.cat(parts, dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, actor_obs, latent=None, motion_embedding=None, **kwargs):
        """Sample action from distribution."""
        self.update_distribution(actor_obs, latent, motion_embedding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs, latent=None, motion_embedding=None):
        """Get deterministic action (mean)."""
        parts = [actor_obs]
        if motion_embedding is not None:
            parts.append(motion_embedding)
        if latent is not None:
            parts.append(latent)
        actor_input = torch.cat(parts, dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')


class PPOCriticPriv(nn.Module):
    """PPO Critic with privileged observation support.
    
    Critic receives critic_obs + priv_latent
    """
    def __init__(self,
                obs_dim_dict,
                module_config_dict):
        super(PPOCriticPriv, self).__init__()

        self.critic_module = BaseModule(obs_dim_dict, module_config_dict)

    @property
    def critic(self):
        return self.critic_module
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, critic_obs, priv_latent=None, motion_embedding=None, **kwargs):
        """Evaluate state value with optional priv_latent and motion_embedding."""
        parts = [critic_obs]
        if priv_latent is not None:
            parts.append(priv_latent)
        if motion_embedding is not None:
            parts.append(motion_embedding)
        critic_input = torch.cat(parts, dim=-1)
        value = self.critic(critic_input)
        return value

