from __future__ import annotations
from copy import deepcopy

from typing import List
import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import BaseModule
from .encoder_modules import ConvEncoder

class PPOActor(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions,
                init_noise_std):
        super(PPOActor, self).__init__()

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

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)

        # Action noise with sigma bounds
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.fix_sigma = module_config_dict.get("fix_sigma", False)
        self.max_sigma = module_config_dict.get("max_sigma", float('inf'))
        self.min_sigma = module_config_dict.get("min_sigma", 0.0)
        
        if self.fix_sigma:
            self.std.requires_grad = False
        
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
    
    @staticmethod
    # not used at the moment
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

    def update_distribution(self, actor_obs, motion_embedding=None):
        if motion_embedding is not None:
            actor_input = torch.cat([actor_obs, motion_embedding], dim=-1)
        else:
            actor_input = actor_obs
        mean = self.actor(actor_input)
        # Clamp std between min_sigma and max_sigma
        std_clamped = (mean * 0. + self.std).clamp(min=self.min_sigma, max=self.max_sigma)
        self.distribution = Normal(mean, std_clamped)

    def act(self, actor_obs, motion_embedding=None, **kwargs):
        self.update_distribution(actor_obs, motion_embedding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs, motion_embedding=None):
        if motion_embedding is not None:
            actor_input = torch.cat([actor_obs, motion_embedding], dim=-1)
        else:
            actor_input = actor_obs
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')

class PPOCritic(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict):
        super(PPOCritic, self).__init__()

        self.critic_module = BaseModule(obs_dim_dict, module_config_dict)

    @property
    def critic(self):
        return self.critic_module
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, critic_obs, motion_embedding=None, **kwargs):
        if motion_embedding is not None:
            critic_input = torch.cat([critic_obs, motion_embedding], dim=-1)
        else:
            critic_input = critic_obs
        value = self.critic(critic_input)
        return value

class PPOActorFixSigma(PPOActor):
    def __init__(self,                 
                 obs_dim_dict,
                network_dict,
                network_load_dict,
                num_actions,):
        super(PPOActorFixSigma, self).__init__(obs_dim_dict, network_dict, network_load_dict, num_actions, 0.0)
        
    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict)['head']
        self.distribution = mean

    @property
    def action_mean(self):
        return self.distribution
    
    def get_actions_log_prob(self, actions):
        raise NotImplementedError
    
    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return self.distribution

