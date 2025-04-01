# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Tuple

import torch.nn as nn
import torch
from torch.distributions.normal import Normal
import numpy as np
import time
import random
import os


def grad_norm(params):
    grad_norm = 0.0
    for p in params:
        if p.grad is not None:
            grad_norm += torch.sum(p.grad**2)
    return torch.sqrt(grad_norm)


def init_mlp(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_activation_func(activation_name):
    if activation_name.lower() == "tanh":
        return nn.Tanh()
    elif activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "elu":
        return nn.ELU()
    elif activation_name.lower() == "identity":
        return nn.Identity()
    else:
        raise NotImplementedError(
            "Actication func {} not defined".format(activation_name)
        )


class ActorDeterministicMLP(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_layers=[64, 64],
        activation="elu",
        device="cuda:0",
    ):
        super(ActorDeterministicMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + hidden_layers + [action_dim]

        def init_(m):
            return init_mlp(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
            )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(get_activation_func(activation))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.actor = nn.Sequential(*modules).to(device)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.actor)

    def get_logstd(self):
        # return self.logstd
        return None

    def forward(self, observations, deterministic=False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_layers=[64, 64],
        logstd_init=-1.0,
        activation="elu",
        device="cuda:0",
    ):
        super(ActorStochasticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + hidden_layers + [action_dim]

        def init_(m):
            return init_mlp(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
            )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(get_activation_func(activation))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))
            else:
                modules.append(get_activation_func("identity"))

        self.mu_net = nn.Sequential(*modules).to(device)

        logstd = logstd_init

        self.logstd = torch.nn.Parameter(
            torch.ones(action_dim, dtype=torch.float32, device=device) * logstd
        )

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.mu_net)
        print(self.logstd)

    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic=False):
        mu = self.mu_net(obs)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp()  # (num_actions)
            # eps = torch.randn((*obs.shape[:-1], std.shape[-1])).to(self.device)
            # sample = mu + eps * std
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample

    def forward_with_dist(self, obs, deterministic=False):
        mu = self.mu_net(obs)
        std = self.logstd.exp()  # (num_actions)

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std

    def evaluate_actions_log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)

        return dist.log_prob(actions)


class CriticMLP(nn.Module):
    def __init__(
        self, obs_dim, hidden_layers=[64, 64], activation="elu", device="cuda:0"
    ):
        super(CriticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + hidden_layers + [1]

        def init_(m):
            return init_mlp(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
            )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(get_activation_func(activation))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = nn.Sequential(*modules).to(device)

        self.obs_dim = obs_dim

        print(self.critic)

    def forward(self, observations):
        return self.critic(observations)


class RunningMeanStd(object):
    def __init__(
        self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device="cuda:0"
    ):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def to(self, device):
        rms = RunningMeanStd(device=device)
        rms.mean = self.mean.to(device).clone()
        rms.var = self.var.to(device).clone()
        rms.count = self.count
        return rms

    @torch.no_grad()
    def update(self, arr: torch.tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0, unbiased=False)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: torch.tensor, batch_var: torch.tensor, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, arr: torch.tensor, un_norm=False) -> torch.tensor:
        if not un_norm:
            result = (arr - self.mean) / torch.sqrt(self.var + 1e-5)
        else:
            result = arr * torch.sqrt(self.var + 1e-5) + self.mean
        return result


class RunningMeanStdObs(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        assert insize is dict
        super(RunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict(
            {
                k: RunningMeanStd(v, epsilon, per_channel, norm_only)
                for k, v in insize.items()
            }
        )

    def forward(self, input, unnorm=False):
        res = {k: self.running_mean_std(v, unnorm) for k, v in input.items()}
        return res


def print_info(*message):
    print("\033[96m", *message, "\033[0m")


class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle=False, drop_last=False):
        self.obs = obs.view(-1, obs.shape[-1])
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()

        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1

    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {
            "obs": self.obs[start_idx:end_idx, :],
            "target_values": self.target_values[start_idx:end_idx],
        }


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.time_total = 0.0

    def on(self):
        assert self.start_time is None, "Timer {} is already turned on!".format(
            self.name
        )
        self.start_time = time.time()

    def off(self):
        assert self.start_time is not None, "Timer {} not started yet!".format(
            self.name
        )
        self.time_total += time.time() - self.start_time
        self.start_time = None

    def report(self):
        print_info(
            "Time report [{}]: {:.2f} seconds".format(self.name, self.time_total)
        )

    def clear(self):
        self.start_time = None
        self.time_total = 0.0


class TimeReport:
    def __init__(self):
        self.timers = {}

    def add_timer(self, name):
        assert name not in self.timers, "Timer {} already exists!".format(name)
        self.timers[name] = Timer(name=name)

    def start_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].on()

    def end_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].off()

    def report(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
        else:
            print_info("------------Time Report------------")
            for timer_name in self.timers.keys():
                self.timers[timer_name].report()
            print_info("-----------------------------------")

    def clear_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].clear()
        else:
            for timer_name in self.timers.keys():
                self.timers[timer_name].clear()

    def pop_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
            del self.timers[name]
        else:
            self.report()
            self.timers = {}


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


def seeding(seed=0, torch_deterministic=False):
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed
