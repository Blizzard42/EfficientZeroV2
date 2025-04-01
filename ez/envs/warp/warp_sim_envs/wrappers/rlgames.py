from typing import List, Optional, Sequence, Union

from warp_sim_envs.envs import Environment, RenderMode
from .utils import cost2reward, eval_timeout
from .utils import get_observation_space, get_action_space
from rl_games.common import env_configurations, vecenv

from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext

import torch

import warp as wp
import numpy as np


class RlgamesEnvWrapper(vecenv.IVecEnv):
    def __init__(
        self,
        env: Environment,
        render_mode: str = None,
        max_episode_length: int = 500,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        image_width: int = 128,
        image_height: int = 128,
        use_graph_capture: bool = False,
        requires_grad: bool = False,
    ):
        self.env = env
        self.env.use_graph_capture = False
        self.use_graph_capture = use_graph_capture

        self.num_envs = self.env.num_envs

        self.render_mode = render_mode

        self.num_obs = self.env.observation_dim
        self.num_actions = self.env.control_dim
        self.device = self.env.device

        self.max_episode_length = max_episode_length

        with wp.ScopedDevice(self.device):
            self.control_gains_wp = wp.array(self.env.control_gains, dtype=wp.float32)
            self.control_limits_wp = wp.array(self.env.control_limits, dtype=wp.float32)
            self.controllable_dofs_wp = wp.array(
                self.env.controllable_dofs, dtype=wp.int32
            )

            self.action_buf = wp.empty(
                (self.num_envs, self.num_actions),
                dtype=wp.float32,
                requires_grad=requires_grad,
            )
            self.rew_buf = wp.empty(
                self.num_envs, dtype=wp.float32, requires_grad=requires_grad
            )
            self.done_buf = wp.empty(self.num_envs, dtype=wp.bool)
            self.episode_length_buf = wp.empty(self.num_envs, dtype=wp.int32)
            self.time_step_buf = wp.empty(self.num_envs, dtype=wp.int32)
            self.timeout_buf = wp.empty(self.num_envs, dtype=wp.bool)
            self.cost_buf = wp.empty(
                self.num_envs, dtype=wp.float32, requires_grad=requires_grad
            )

        self.step_count = 0
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias

        self.graph = None
        # if self.use_graph_capture:
        #     with wp.ScopedCapture() as capture:
        #         self.env.update()
        #     self.graph = capture.graph

        self.extras = {}
        self.obs_dict = {}

        self.render_mode = render_mode
        self.env.render_mode = RenderMode.NONE
        if render_mode == "human" or render_mode == "rgb_array":
            self.env.render_mode = RenderMode.OPENGL
        if render_mode == "rgb_array":
            self.env.use_tiled_rendering = True
            self.env.opengl_render_settings = dict(scaling=3.0, draw_axis=False)
            self.env.opengl_tile_render_settings = dict(
                tile_width=image_width, tile_height=image_height
            )
            self.compute_observations = self.compute_rgb_observations
            self.obs_buf = wp.empty(
                (self.num_envs, image_height, image_width, 3),
                dtype=wp.uint8,
                device=self.device,
                # do not require grads here since we only return it during env.reset()
                requires_grad=False,
            )
        else:
            self.compute_observations = self.compute_state_observations
            self.obs_buf = wp.empty(
                (self.num_envs, self.num_obs),
                dtype=wp.float32,
                device=self.device,
                # do not require grads here since we only return it during env.reset()
                requires_grad=False,
            )
        self.env.setup_renderer()
        self.observation_space = get_observation_space(
            self.env, render_mode, use_gymnasium=False
        )
        self.action_space = get_action_space(self.env, use_gymnasium=False)
        self.requires_grad = requires_grad

    def compute_state_observations(self, state, control, obs_buf):
        self.env.compute_observations(
            state,
            control,
            obs_buf,
            0,
            0,
        )

    def compute_rgb_observations(self, state, control, obs_buf):
        self.env.render(state=state)
        self.env.renderer.get_pixels(
            obs_buf, split_up_tiles=True, mode="rgb", use_uint8=True
        )

    def reset(self):
        self.env.reset()
        self.step_count = 0
        self.done_buf.fill_(0)
        self.episode_length_buf.fill_(0)
        self.compute_observations(self.env.state, self.env.control, self.obs_buf)
        return wp.to_torch(self.obs_buf)

    def render(self, state=None):
        self.env.render(state=state)

    def step_internal(self, act: wp.array):
        # inner step function, to be captured in a CUDA graph
        control = self.env.control
        self.control = control
        self.env.assign_control(act, control, self.env.state)

        if not self.requires_grad:
            state = self.env.update()
        else:
            for i in range(self.env.sim_substeps):
                self.env.step(
                    self.env.states[self.env.sim_step],
                    self.env.states[self.env.sim_step + 1],
                    control,
                    eval_collisions=False,
                )
            state = self.env.states[self.env.sim_substeps]
        # print([state.joint_q.ptr for state in self.env.states])
        # print(state.joint_q.ptr)

        # reward is single-step, not cumulative
        self.cost_buf.zero_()
        self.done_buf.zero_()
        self.env.compute_cost_termination(
            state,
            control,
            self.step_count,
            self.max_episode_length,
            self.cost_buf,
            self.done_buf,
        )
        # wp.copy(self.done_buf_cpu, self.done_buf)
        # terminated = self.step_count >= self.max_episode_length or self.has_terminated

        wp.launch(
            eval_timeout,
            dim=self.num_envs,
            inputs=[
                self.max_episode_length,
            ],
            outputs=[
                self.timeout_buf,
                self.episode_length_buf,
                self.done_buf,
            ],
            device=self.device,
        )

        wp.launch(
            cost2reward,
            dim=self.num_envs,
            inputs=[
                self.cost_buf,
                self.reward_scale,
                self.reward_bias,
            ],
            outputs=[self.rew_buf],
            device=self.device,
        )
        # wp.copy(self.rew_buf_cpu, self.rew_buf)
        self.compute_observations(state, control, self.obs_buf)

        self.env.reset_envs(self.done_buf, state=state)

        self.state = state

        # return (potentially) differentiable observations and rewards
        return self.obs_buf, self.rew_buf

    def step(self, actions: torch.Tensor):
        """
        Step the environments with the given action

        :param actions: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        act = wp.from_torch(actions)

        if self.use_graph_capture:
            # assign to a fixed action buffer
            self.action_buf.assign(act)
            if self.graph is None:
                with wp.ScopedCapture() as capture:
                    self.step_internal(self.action_buf)
                self.graph = capture.graph
            else:
                wp.capture_launch(self.graph)
        else:
            self.step_internal(act)

        # tape.visualize(
        #     filename="tape.dot",
        #     track_inputs=[act],
        #     track_input_names=["act"],
        #     track_outputs=[self.rew_buf, self.obs_buf],
        #     track_output_names=["rew_buf", "obs_buf"],
        # )

        self.step_count += 1

        obs = wp.to_torch(self.obs_buf)
        rewards = wp.to_torch(self.rew_buf)
        dones = wp.to_torch(self.done_buf)

        invalid = torch.isnan(rewards) | torch.any(torch.isnan(obs), dim=1)
        if invalid.any():
            rewards[invalid] = -1000.0
            self.env.reset_envs(wp.from_torch(invalid), state=self.state)
            self.compute_observations(self.state, self.control, self.obs_buf)
            obs = wp.to_torch(self.obs_buf)
            dones = dones | invalid
        if torch.isnan(rewards).any() or torch.any(torch.isnan(obs), dim=1).any() or torch.isnan(dones).any() or torch.isnan(actions).any():
            raise ValueError("Nan in rewards or observations")

        # copy warp data to pytorch
        self.extras["time_outs"] = wp.to_torch(self.timeout_buf).squeeze(-1)
        self.obs_dict["obs"] = obs

        self.render(state=self.state)

        return self.obs_dict, rewards, dones, self.extras

    def close(self):
        """
        Clean up the environment's resources.
        """
        if self.env.renderer:
            self.env.renderer.close()

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        if attr_name == "render_mode":
            return [self.render_mode for _ in range(self.num_envs)]
        return getattr(self.env, attr_name)

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        return getattr(self.env, method_name)(*method_args, **method_kwargs)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: (Optional[int]) The random seed. May be None for completely random seeding.
        :return: (List[Union[None, int]]) Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        pass

    def get_images(self) -> Sequence[np.ndarray]:
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def env_is_wrapped(self) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper

        :return: (List[bool]) List of bools for each environment, indicating if wrapped
        """
        raise NotImplementedError

    def get_env_info(self):
        info = {}
        info["action_space"] = self.action_space
        info["observation_space"] = self.observation_space
        return info

    def get_number_of_agents(self):
        return 1  # only used for multi-agent environments
    
    @property
    def name(self):
        return self.env.sim_name


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats."""

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(
            self.algo.ppo_device
        )
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if "episode" in infos:
                self.ep_infos.append(infos["episode"])

            if len(infos) > 0 and isinstance(
                infos, dict
            ):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if (
                        isinstance(v, float)
                        or isinstance(v, int)
                        or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                    ):
                        self.direct_info[k] = v

    def after_clear_stats(self):
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.algo.device))
                    )
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, epoch_num)
            self.ep_infos.clear()

        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, frame)
            self.writer.add_scalar(f"{k}/iter", v, epoch_num)
            self.writer.add_scalar(f"{k}/time", v, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar("scores/mean", mean_scores, frame)
            self.writer.add_scalar("scores/iter", mean_scores, epoch_num)
            self.writer.add_scalar("scores/time", mean_scores, total_time)


def register_env(
    env: Environment,
    render_mode: str = "human",
    max_episode_length: int = 500,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    image_width: int = 128,
    image_height: int = 128,
    env_name: str = "warp",
    use_graph_capture: bool = True,
    requires_grad: bool = False,
):
    env = RlgamesEnvWrapper(
        env,
        render_mode=render_mode,
        max_episode_length=max_episode_length,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        image_width=image_width,
        image_height=image_height,
        use_graph_capture=use_graph_capture,
        requires_grad=requires_grad,
    )

    vecenv.register(
        "WARP",
        lambda config_name, num_actors, **kwargs: env,
    )
    env_configurations.register(
        env_name,
        {
            "vecenv_type": "WARP",
            "env_creator": lambda **_: env,
        },
    )


# class MonitoringWrapper(RlgamesEnvWrapper):
#     def __init__(self, env, algo_observer, **kwargs):
#         self.env = env
#         self.algo_observer = algo_observer
#         super().__init__(**kwargs)

#     def render(self):
#         super().render()


# class TinyRendererMonitoringWrapper(MonitoringWrapper):
#     def __init__(self, env, algo_observer, **kwargs):
#         super().__init__(env, algo_observer, **kwargs)

#     def render(self):
#         # update human renderer given the current state
#         # Warp array of warp.transform for the rigid bodies
#         self.env.state.body_q

#         super().render()

# class ObservationWrapper(RlgamesEnvWrapper):
#     def __init__(self, env, algo_observer, **kwargs):
#         self.env = env
#         self.algo_observer = algo_observer
#         super().__init__(**kwargs)
#         self.observation_space = gym.spaces.Box(...)

#     def compute_state_observations(self):
#         # update synthetic camera renderer given the current state
#         # Warp array of warp.transform
#         self.env.state.body_q
#         # get pixels
#         return torch.tensor(....)
