# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
# Adapted for WARP Hand Environment
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import time
import copy
import math
import torch
import torchrl
import torch.nn as nn
from ez.agents.base import Agent
from omegaconf import open_dict

import numpy as np
# --- CHANGE 1: Import the correct environment creation function ---
# from ez.envs import make_dmc
from ez.envs import make_warp_hand # Or potentially just 'import gymnasium' if using make directly
# --- End CHANGE 1 ---

from ez.utils.format import DiscreteSupport
from ez.agents.models import EfficientZero # Assuming EfficientZero model class is general

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# --- NOTE: Network definitions (mlp, ImproveResidualBlock, PNorm,        ---
# --- RunningMeanStd, RepresentationNetwork, DynamicsNetwork, RewardNetwork ---
# --- ValuePolicyNetwork) are kept IDENTICAL to ez_dmc_state.py as they    ---
# --- seem generally applicable to state-based continuous control.        ---
# --- If performance issues arise, these might need tuning later.         ---

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    init_zero=False,
    use_bn=True,
    p_norm=False,
    noisy=False
):
    """MLP layers - Copied from ez_dmc_state.py"""
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            if use_bn:
                layers += [
                    torchrl.modules.NoisyLinear(sizes[i], sizes[i + 1], std_init=0.5) if noisy else nn.Linear(sizes[i], sizes[i + 1]),
                    nn.BatchNorm1d(sizes[i + 1]),
                    act()
                ]
            else:
                layers += [torchrl.modules.NoisyLinear(sizes[i], sizes[i + 1], std_init=0.5) if noisy else nn.Linear(sizes[i], sizes[i + 1]),
                           act()]
        else:
            if p_norm == True:
                layers += [PNorm()]
            act = output_activation
            layers += [torchrl.modules.NoisyLinear(sizes[i], sizes[i + 1], std_init=0.5) if noisy else nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        if noisy:
            layers[-2].reset_parameters()
        else:
            layers[-2].weight.data.fill_(0)
            layers[-2].bias.data.fill_(0)


    return nn.Sequential(*layers)


class ImproveResidualBlock(nn.Module):
    """Copied from ez_dmc_state.py"""
    def __init__(self, input_shape, hidden_shape):
        super(ImproveResidualBlock, self).__init__()
        self.ln1 = nn.LayerNorm(input_shape)
        self.linear1 = nn.Linear(input_shape, hidden_shape)
        self.linear2 = nn.Linear(hidden_shape, input_shape)

    def forward(self, x):
        identity = x
        out = self.ln1(x)
        out = self.linear1(out)
        out = nn.functional.relu(out)
        out = self.linear2(out)

        out += identity
        return out


class PNorm(nn.Module):
    """Copied from ez_dmc_state.py"""
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        assert len(x.shape) == 2
        return nn.functional.normalize(x, dim=1, eps=self.eps)


class RunningMeanStd(nn.Module):
    """Copied from ez_dmc_state.py"""
    def __init__(self, shape, epsilon=1e-5, momentum=0.1):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.count = 1e3
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]
            self.running_mean, self.running_var, self.count = self.update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, mean, var, batch_count)
            global_mean = self.running_mean
            global_var = self.running_var
        else:
            global_mean = self.running_mean
            global_var = self.running_var
        # Ensure dimensions match for broadcasting if x is 1D
        if x.ndim == 1 and global_mean.ndim == 1 and x.shape == global_mean.shape:
             x = (x - global_mean) / torch.sqrt(global_var + self.epsilon)
        elif x.ndim > 1 :
             x = (x - global_mean) / torch.sqrt(global_var + self.epsilon)
        else:
            # Handle potential shape mismatch or scalar case if necessary
             print(f"Warning: RunningMeanStd shape mismatch or unexpected input dim. x:{x.shape}, mean:{global_mean.shape}")
             # Fallback or error handling might be needed
        return x


    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

class RepresentationNetwork(nn.Module):
    """Representation network - Copied from ez_dmc_state.py"""
    def __init__(
        self,
        observation_shape, # This will be a single integer for state vectors
        n_stacked_obs, # Likely 1 for state vectors
        num_blocks,
        rep_net_shape,
        hidden_shape,
        use_bn=True,
    ):
        super().__init__()
        # For state-based, observation_shape is just the dimension (int)
        state_dim = observation_shape * n_stacked_obs
        self.running_mean_std = RunningMeanStd(state_dim)
        self.mlp = nn.Linear(state_dim, hidden_shape)
        self.ln = nn.LayerNorm(hidden_shape)
        self.Rep_resblocks = nn.ModuleList(
            [ImproveResidualBlock(hidden_shape, rep_net_shape) for _ in range(num_blocks)]
        )

    def forward(self, x):
        # If input is [batch_size, state_dim]
        if x.ndim == 1: # Handle case where batch size might be 1 and squeezed
             x = x.unsqueeze(0)
        x = self.running_mean_std(x)
        x = self.mlp(x)
        x = self.ln(x)
        x = torch.tanh(x) # Changed from relu? Seems like a reasonable choice.
        for block in self.Rep_resblocks:
            x = block(x)
        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean) if mean else 0.0
        return mean


class DynamicsNetwork(nn.Module):
    """Dynamics network - Copied from ez_dmc_state.py"""
    def __init__(
        self,
        hidden_shape,
        action_shape,
        num_blocks,
        dyn_shape,
        act_embed_shape,
        rew_net_shape, # Note: This seems unused here, reward handled separately
        reward_support_size, # Note: This seems unused here, reward handled separately
        init_zero=False, # Note: This seems unused here
        use_bn=True, # Note: BN used internally in MLP? But not explicit here.
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.act_linear1 = nn.Linear(action_shape, act_embed_shape)
        self.act_ln1 = nn.LayerNorm(act_embed_shape)
        self.dyn_ln_1 = nn.LayerNorm(hidden_shape + act_embed_shape)
        self.dyn_net_1 = nn.Linear(hidden_shape + act_embed_shape, dyn_shape)
        self.dyn_ln_2 = nn.LayerNorm(dyn_shape) # This LN seems misplaced, maybe should be before dyn_net_2? Based on ResBlock structure.
        self.dyn_net_2 = nn.Linear(dyn_shape, hidden_shape)

        if num_blocks > 0:
            self.dyn_resblocks = nn.ModuleList(
                [ImproveResidualBlock(hidden_shape, dyn_shape) for _ in range(num_blocks)]
            )
        else:
            self.dyn_resblocks = nn.ModuleList([])

    def forward(self, hidden, action, reward_hidden=None): # reward_hidden unused
        act_emb = self.act_linear1(action)
        act_emb = self.act_ln1(act_emb)
        act_emb = nn.functional.relu(act_emb)

        x = self.dyn_ln_1(torch.cat((hidden, act_emb), dim=-1))
        x = self.dyn_net_1(x)
        x = nn.functional.relu(x)
        # Applying LayerNorm here seems inconsistent with standard ResNet pre-activation
        # x = self.dyn_ln_2(x)
        x = self.dyn_net_2(x)
        state = hidden + x # Residual connection

        for block in self.dyn_resblocks:
            state = block(state)

        return state

    def get_dynamic_mean(self):
        mean = []
        for name, param in self.dyn_net_1.named_parameters():
             if param.requires_grad: # Only consider trainable parameters
                mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean) if mean else 0.0
        return mean


class RewardNetwork(nn.Module):
    """Copied from ez_dmc_state.py"""
    def __init__(
        self,
        hidden_shape,
        rew_net_shape,
        reward_support_size,
        init_zero=False,
        use_bn=True,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.rew_net_shape = rew_net_shape
        self.reward_support_size = reward_support_size
        # Applying ResBlock/LN before MLP prediction head
        self.rew_resblock = ImproveResidualBlock(self.hidden_shape, self.hidden_shape)
        self.ln = nn.LayerNorm(self.hidden_shape)
        self.rew_net = mlp(self.hidden_shape, self.rew_net_shape, self.reward_support_size,
                           init_zero=init_zero,
                           use_bn=use_bn)

    def forward(self, next_state):
        next_state = self.rew_resblock(next_state)
        next_state = self.ln(next_state)
        reward = self.rew_net(next_state)
        return reward


class RewardNetworkLSTM(nn.Module):
    """Copied from ez_dmc_state.py"""
    def __init__(
        self,
        hidden_shape,
        rew_net_shape,
        reward_support_size,
        lstm_hidden_size,
        init_zero=False,
        use_bn=True,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.rew_net_shape = rew_net_shape
        self.reward_support_size = reward_support_size
        self.rew_resblock = ImproveResidualBlock(self.hidden_shape, self.hidden_shape)
        self.ln = nn.LayerNorm(self.hidden_shape)
        self.lstm = nn.LSTM(input_size=self.hidden_shape, hidden_size=lstm_hidden_size)
        self.rew_net = mlp(lstm_hidden_size, self.rew_net_shape, self.reward_support_size,
                           init_zero=init_zero,
                           use_bn=use_bn)

    def forward(self, next_state, hidden):
        next_state = self.rew_resblock(next_state)
        next_state = self.ln(next_state)
        next_state = next_state.unsqueeze(0) # Add sequence dimension for LSTM
        reward, hidden = self.lstm(next_state, hidden)
        reward = reward.squeeze(0) # Remove sequence dimension
        reward = self.rew_net(reward)
        return reward, hidden


class ValuePolicyNetwork(nn.Module):
    """Predict value and policy - Copied from ez_dmc_state.py"""
    def __init__(
        self,
        hidden_shape,
        val_net_shape,
        pi_net_shape,
        action_shape,
        full_support_size, # This is the value support size
        init_zero=False,
        use_bn=True,
        p_norm=False,
        policy_distr='squashed_gaussian',
        noisy=False,
        value_support=None, # Config detail, not network structure
        **kwargs
    ):
        super().__init__()
        self.v_num = kwargs.get('v_num', 1) # Default to 1 if not specified
        self.hidden_shape = hidden_shape
        self.val_net_shape = val_net_shape
        self.action_shape = action_shape
        self.pi_net_shape = pi_net_shape

        self.action_space_size = action_shape # This is potentially confusing, action_shape is dim
        # Parameters for squashed_gaussian, defaults from DMC agent
        self.init_std = kwargs.get('init_std', 1.0) # Allow overriding via kwargs
        self.min_std = kwargs.get('min_std', 0.1)
        self.policy_distr = policy_distr

        # Value head processing
        self.val_resblock = ImproveResidualBlock(hidden_shape, hidden_shape)
        self.val_ln = nn.LayerNorm(hidden_shape)
        self.val_nets = nn.ModuleList([
            mlp(self.hidden_shape, self.val_net_shape, full_support_size, use_bn=use_bn)
            for _ in range(self.v_num)
        ])

        # Policy head processing
        self.pi_resblock = ImproveResidualBlock(hidden_shape, hidden_shape)
        self.pi_ln = nn.LayerNorm(hidden_shape)
        # Output size is action_dim * 2 (for mean and std dev)
        self.pi_net = mlp(self.hidden_shape, self.pi_net_shape, self.action_shape * 2,
                          init_zero=init_zero,
                          use_bn=use_bn,
                          p_norm=p_norm, # Applied before output activation in mlp helper
                          noisy=noisy)

        self.noisy = noisy
        # self.value_support = value_support # Not used in forward/backward directly

    def reset_noise(self):
        if self.noisy:
            for layer in self.pi_net:
                try:
                    layer.reset_noise()
                except AttributeError: # Handle non-noisy layers in Sequential
                    pass

    def forward(self, x):
        # Value prediction
        value = self.val_resblock(x)
        value = self.val_ln(value)
        values = []
        for val_net in self.val_nets:
            values.append(val_net(value))
        values = torch.stack(values) # Shape: [v_num, batch_size, value_support_size]

        # Policy prediction
        policy_params = self.pi_resblock(x)
        policy_params = self.pi_ln(policy_params)
        policy_params = self.pi_net(policy_params) # Shape: [batch_size, action_shape * 2]

        # Post-process policy parameters based on distribution type
        mean, std_param = torch.chunk(policy_params, 2, dim=-1)
        if self.policy_distr == 'squashed_gaussian':
            mean = 5 * torch.tanh(mean / 5) # Soft clamp mu, range approx [-5, 5]
            std = torch.nn.functional.softplus(std_param + self.init_std) + self.min_std
        elif self.policy_distr == 'truncated_gaussian': # Example if needed
            std = torch.nn.functional.softplus(std_param) + self.min_std
            # Clamping or truncation happens when sampling/calculating log_prob
        else: # Default to gaussian-like std dev calculation
            std = torch.nn.functional.softplus(std_param) + self.min_std

        # Return value logits and policy parameters (mean, std)
        # Returning std directly is often more convenient than the raw parameter
        policy_out = torch.cat([mean, std], dim=-1)

        return values, policy_out

    def log_std(self, x, low, dif):
        # This seems like an alternative std parameterization, not used by default?
        # Keeping it for potential future use or if config changes policy_distr
        return low + 0.5 * dif * (torch.tanh(x) + 1)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# --- End of Network Definitions ---


# --- CHANGE 2: Rename Agent Class ---
class WARPStateAgent(Agent):
# --- End CHANGE 2 ---
    def __init__(self, config):
        super().__init__(config)

        self.update_config() # Call config update after super init

        # Store config values needed by build_model (copied from DMC agent)
        self.num_blocks = config.model.num_blocks
        # self.fc_layers = config.model.fc_layers # fc_layers seems unused in DMC build_model?
        # self.down_sample = config.model.down_sample # Not relevant for state
        self.state_norm = config.model.state_norm
        self.value_prefix = config.model.value_prefix
        self.init_zero = config.model.init_zero
        # self.value_ensumble = config.model.value_ensumble # v_num is used instead
        self.value_policy_detach = config.train.value_policy_detach
        self.v_num = config.train.v_num

    def update_config(self):
        # Prevent multiple updates if called accidentally
        if hasattr(self, '_update') and self._update:
             print("Warning: update_config called multiple times.")
             # return # Allow re-running for interactive debugging if needed

        # --- CHANGE 3: Instantiate the correct environment ---
        print("Instantiating WARP environment to get specs...")
        # Pass kwargs from the config's env section to the make function
        env = make_warp_hand(self.config.env.game, seed=0, save_path=None, **self.config.env)
        # --- End CHANGE 3 ---

        # Get observation and action space info from the instantiated env
        # Assuming state-based observation, shape is a tuple like (dim,)
        obs_shape = env.observation_space.shape[0] # Get the dimension
        action_space_size = env.action_space.shape[0]
        print(f"Detected Obs Shape: {obs_shape}, Action Space Size: {action_space_size}")

        # --- CHANGE 4: Close the temporary env ---
        # Important to release resources, especially if WARP uses GPU memory
        env.close()
        del env
        # --- End CHANGE 4 ---

        # Calculate reward/value support sizes (logic copied from DMC agent)
        # Ensure DiscreteSupport uses the correct config sections
        reward_support = DiscreteSupport(self.config) 
        reward_size = reward_support.size
        self.reward_support = reward_support # Store for potential use

        value_support = DiscreteSupport(self.config)
        value_size = value_support.size
        self.value_support = value_support # Store for potential use

        # Update the config object (OmegaConf)
        localtime = time.strftime('%Y-%m-%d %H:%M:%S')
        # --- CHANGE 5: Update tag generation if desired ---
        # Use env.env (e.g., 'WARPHand') and env.game for tag
        tag = f'{self.config.env.env}-{self.config.env.game}-seed={self.config.env.base_seed}-{localtime}/'
        # --- End CHANGE 5 ---

        with open_dict(self.config):
            self.config.env.action_space_size = action_space_size
            self.config.env.obs_shape = obs_shape # Store the integer dim
            # Discount factor adjustment based on frame skip (if applicable)
            # self.config.rl.discount **= self.config.env.n_skip # Keep if frame skip > 1
            self.config.model.reward_support.size = reward_size
            self.config.model.value_support.size = value_size
            # Update save path - ensure 'save_path' exists in the base config
            if 'save_path' not in self.config:
                 self.config.save_path = './outputs/' # Default save path if not set
            self.config.save_path += tag

        # Store key dimensions needed by build_model (copied from DMC agent)
        self.action_space_size = self.config.env.action_space_size
        self.obs_shape = self.config.env.obs_shape
        self.n_stack = self.config.env.n_stack # Should be 1 for state
        self.rep_net_shape = self.config.model.rep_net_shape
        self.hidden_shape = self.config.model.hidden_shape
        self.dyn_shape = self.config.model.dyn_shape
        self.act_embed_shape = self.config.model.act_embed_shape
        self.rew_net_shape = self.config.model.rew_net_shape
        self.val_net_shape = self.config.model.val_net_shape
        self.pi_net_shape = self.config.model.pi_net_shape

        self.proj_hid_shape = self.config.model.proj_hid_shape
        self.pred_hid_shape = self.config.model.pred_hid_shape
        self.proj_shape = self.config.model.proj_shape
        self.pred_shape = self.config.model.pred_shape

        self._update = True # Mark config as updated
        self.use_bn = self.config.model.use_bn
        self.use_p_norm = self.config.model.use_p_norm
        self.noisy_net = self.config.model.noisy_net

        print("WARPStateAgent configuration updated.")
        print(f"  Action Space Size: {self.action_space_size}")
        print(f"  Observation Shape: {self.obs_shape}")


    def build_model(self):
        """Builds the EfficientZero model using configured dimensions."""

        if not hasattr(self, '_update') or not self._update:
             raise RuntimeError("Agent config not updated before calling build_model. Call update_config first.")

        print("Building WARPStateAgent model...")
        # --- CHANGE 6: Ensure correct dimensions are passed ---
        # Representation network expects obs_shape (int) and n_stack (int)
        representation_model = RepresentationNetwork(self.obs_shape, self.n_stack, self.num_blocks,
                                                     self.rep_net_shape, self.hidden_shape,
                                                     use_bn=self.use_bn)
        # --- End CHANGE 6 ---

        # Determine output sizes for reward/value based on support type
        # (Logic copied from DMC agent)
        value_output_size = self.config.model.value_support.size if self.config.model.value_support.type != 'symlog' else 1
        reward_output_size = self.config.model.reward_support.size if self.config.model.reward_support.type != 'symlog' else 1

        # Instantiate other network components (using stored dimensions)
        dynamics_model = DynamicsNetwork(self.hidden_shape, self.action_space_size, self.num_blocks, self.dyn_shape,
                                         self.act_embed_shape, self.rew_net_shape, reward_output_size, # reward params unused here
                                         use_bn=self.use_bn)

        value_policy_model = ValuePolicyNetwork(self.hidden_shape, self.val_net_shape, self.pi_net_shape,
                                                self.action_space_size, value_output_size,
                                                init_zero=self.init_zero, use_bn=self.use_bn, p_norm=self.use_p_norm,
                                                policy_distr=self.config.model.policy_distribution, noisy=self.noisy_net,
                                                value_support=self.config.model.value_support, # Pass config section
                                                v_num=self.v_num, # Pass v_num
                                                # Pass std params if needed by ValuePolicyNetwork
                                                init_std=self.config.model.get('init_std', 1.0),
                                                min_std=self.config.model.get('min_std', 0.1)
                                                )


        # Conditional reward network (copied from DMC agent)
        if self.config.model.value_prefix: # Assuming value_prefix means use LSTM reward net
            reward_prediction_model = RewardNetworkLSTM(self.hidden_shape, self.rew_net_shape, reward_output_size, self.config.model.lstm_hidden_size,
                                                        init_zero=self.init_zero, use_bn=self.use_bn)
        else:
            reward_prediction_model = RewardNetwork(self.hidden_shape, self.rew_net_shape, reward_output_size,
                                                    init_zero=self.init_zero, use_bn=self.use_bn)

        # Projection networks (copied from DMC agent)
        projection_model = nn.Sequential(
            nn.Linear(self.hidden_shape, self.proj_hid_shape),
            nn.LayerNorm(self.proj_hid_shape),
            nn.ReLU(),
            nn.Linear(self.proj_hid_shape, self.proj_hid_shape),
            nn.LayerNorm(self.proj_hid_shape),
            nn.ReLU(),
            nn.Linear(self.proj_hid_shape, self.proj_shape),
            nn.LayerNorm(self.proj_shape)
        )
        projection_head_model = nn.Sequential(
            nn.Linear(self.proj_shape, self.pred_hid_shape),
            nn.LayerNorm(self.pred_hid_shape),
            nn.ReLU(),
            nn.Linear(self.pred_hid_shape, self.pred_shape),
        )

        # Assemble the final EfficientZero model
        ez_model = EfficientZero(representation_model, dynamics_model, reward_prediction_model, value_policy_model,
                                 projection_model, projection_head_model, self.config,
                                 state_norm=self.state_norm, value_prefix=self.value_prefix)

        print("Model built successfully.")
        return ez_model

# --- CHANGE 7: Register the new agent (if needed by the framework) ---
# This step depends on how agents are loaded in your __main__ or train.py
# For example, if there's a dictionary like `agents.names`:
# In ez/agents/__init__.py:
# from .ez_warp_state import WARPStateAgent
# names = { ... 'warp_state_agent': WARPStateAgent, ... }
# Then set agent_name: warp_state_agent in the config file.
# --- End CHANGE 7 ---