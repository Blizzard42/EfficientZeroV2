from rsl_rl.env import VecEnv

import warp as wp
import numpy as np
import torch

from typing import Tuple

from warp_sim_envs.envs import Environment
from .utils import assign_controls, reset_envs




def make_rsl_env(
    Env,
    traj_length: int = 500,
    num_envs: int = 100,
    env_args={},
    render_mode=None,
    clear_nans=True,
    add_noise=False,
    noise_scale_vec=0.01,
):
    class RslEnv(VecEnv):
        """RslEnv class for a vectorized RSL RL gym environment."""

        # num_envs: int
        """Number of environments."""
        # num_obs: int
        """Number of observations."""
        # num_privileged_obs: int
        """Number of privileged observations."""
        # num_actions: int
        """Number of actions."""
        # max_episode_length: int
        """Maximum episode length."""
        # privileged_obs_buf: torch.Tensor
        """Buffer for privileged observations."""
        # obs_buf: torch.Tensor
        """Buffer for observations."""
        # rew_buf: torch.Tensor
        """Buffer for rewards."""
        # reset_buf: torch.Tensor
        """Buffer for resets."""
        # episode_length_buf: torch.Tensor  # current episode duration
        """Buffer for current episode lengths."""
        # extras: dict
        """Extra information (metrics).

        Extra information is stored in a dictionary. This includes metrics such as the episode reward, episode length,
        etc. Additional information can be stored in the dictionary such as observations for the critic network, etc.
        """
        # device: torch.device
        """Device to use."""

        """
        Operations.
        """

        def __init__(self, num_envs: int):

            self.num_envs = num_envs
            self.traj_length = traj_length
            self.render_mode = render_mode
            self.clear_nans = clear_nans
            self.add_noise = add_noise
            self.noise_scale_vec = noise_scale_vec

            env_args["num_envs"] = num_envs
            self.env: Environment = Env(**env_args)

            if self.env.use_graph_capture:
                with wp.ScopedCapture() as capture:
                    self.env.update()
                self.graph = capture.graph
            else:
                self.graph = None

            self.control_gains_wp = wp.array(self.env.control_gains, dtype=wp.float32)
            self.control_limits_wp = wp.array(self.env.control_limits, dtype=wp.float32)
            self.controllable_dofs_wp = wp.array(self.env.controllable_dofs, dtype=wp.int32)
            self.control_full_dim = len(self.env.control_input) // self.num_envs

            self.num_obs = self.env.observation_dim
            self.num_actions = self.env.control_dim
            self.num_privileged_obs = 0
            self.max_episode_length = traj_length
            self.privileged_obs_buf = None
            self.device = "cuda"

            # allocate buffers
            self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
            self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
            self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.time_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

            self._cost_buffer = wp.zeros(self.num_envs, dtype=wp.float32)

            # if self.num_privileged_obs is not None:
            #     self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
            # else:
            #     self.privileged_obs_buf = None
            # pass

            self.extras = {}
            self.extras["observations"] = {}

            self._step_count = 0

        def get_observations(self) -> Tuple[torch.Tensor, dict]:
            """Return the current observations.

            Returns:
                Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
            """

            self.env.compute_observations(
                self.env.state,
                self.env.control,
                wp.from_torch(self.obs_buf),
                0,
                0,
            )

            # wp.launch(
            #     eval_observations,
            #     dim=self.num_envs,
            #     inputs=[
            #         self.sim.env_cartpole.state.joint_q,
            #         self.sim.env_cartpole.state.joint_qd,
            #     ],
            #     outputs=[wp.from_torch(self.obs_buf)],
            # )

            if self.clear_nans:
                self.obs_buf[torch.isnan(self.obs_buf)] = 0.0

            # add noise if needed
            if self.add_noise:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

            return self.obs_buf, self.extras

        def reset(self) -> Tuple[torch.Tensor, dict]:
            """Reset all environment instances.

            Returns:
                Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
            """

            self.env.reset()
            self._step_count = 0
            self.done_buf.fill_(0)
            self.reset_buf.fill_(0)
            self.episode_length_buf.fill_(0)

            #  if self._discrete_actions:
            #   force = self.force_mag if action == 1 else -self.force_mag
            # else:
            #   force = action[0]

            # p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=force)
            # p.stepSimulation()

            # self.state = p.getJointState(self.cartpole, 1)[0:2] + p.getJointState(self.cartpole, 0)[0:2]
            # theta, theta_dot, x, x_dot = self.state

            # done =  x < -self.x_threshold \
            #             or x > self.x_threshold \
            #             or theta < -self.theta_threshold_radians \
            #             or theta > self.theta_threshold_radians
            # done = bool(done)
            # reward = 1.0
            # #print("state=",self.state)
            # return np.array(self.state), reward, done, {}

            # self.sim.env_cartpole.reset()
            return self.get_observations()

        def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
            """Apply input action on the environment.

            Args:
                actions (torch.Tensor): Input actions to apply. Shape: (num_envs, num_actions)

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
                    A tuple containing the observations, rewards, dones and extra information (metrics).
            """

            self.episode_length_buf[self.done_buf] = 0
            self.done_buf.fill_(0)
            self.reset_buf.fill_(0)

            # TODO see https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py

            # self.env.control_input.assign(actions * self.env.control_gains)
            wp.launch(
                assign_controls,
                dim=self.num_envs,
                inputs=[
                    wp.from_torch(actions),
                    self.control_gains_wp,
                    self.control_limits_wp,
                    self.controllable_dofs_wp,
                    self.control_full_dim,
                ],
                outputs=[
                    self.env.control_input,
                ],
                device=self.env.device,
            )
            if self.env.use_graph_capture:
                wp.capture_launch(self.graph)
            else:
                self.env.update()

            # reward is single-step, not cumulative
            self._cost_buffer.zero_()
            self.env.compute_cost_termination(
                self.env.state,
                self.env.control,
                self._step_count,
                self.traj_length,
                self._cost_buffer,
                wp.from_torch(self.done_buf),
            )
            # terminated = self._step_count >= self.traj_length or self.has_terminated
            # wp.copy(self._cost_buffer_cpu, self._cost_buffer)
            reward = -wp.to_torch(self._cost_buffer) + 10.0
            obs_buf, extras = self.get_observations()

            self.env.render()

            self.reset_idx(self.done_buf)

            self._step_count += 1

            return obs_buf, reward, self.done_buf, extras

        def reset_idx(self, reset_buf):
            reset_envs(self.env, wp.from_torch(reset_buf))

            self.episode_length_buf[reset_buf] = 0
            # self.done_buf[reset_buf] = 1

            # wp.launch(
            #     apply_actions,
            #     dim=self.num_envs,
            #     inputs=[wp.from_torch(actions)],
            #     outputs=[self.sim.env_cartpole.control.joint_act],
            # )

            # self.sim.step()
            # # A tuple containing the observations, rewards, dones and extra information (metrics).

            # wp.launch(
            #     eval_rewards,
            #     dim=self.num_envs,
            #     inputs=[
            #         self.sim.env_cartpole.state.joint_q,
            #         self.sim.env_cartpole.state.joint_qd,
            #         self.sim.env_cartpole.control.joint_act,
            #         wp.from_torch(self.time_step_buf),
            #         self.max_episode_length,
            #         self.sim.env_cartpole.model.joint_q,
            #         self.sim.env_cartpole.model.joint_qd,
            #     ],
            #     outputs=[
            #         wp.from_torch(self.rew_buf),
            #         wp.from_torch(self.done_buf),
            #     ],
            # )
            # self.get_observations()

            # return self.obs_buf, self.rew_buf, self.done_buf, self.extras

        # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

        # def __init__(self, render_mode=None):
        #     env_render_mode = RenderMode.NONE
        #     if render_mode == "human" or render_mode == "rgb_array":
        #         env_render_mode = RenderMode.OPENGL
        #     env_args["render_mode"] = env_render_mode
        #     env_args["num_envs"] = 1
        #     self.env: Environment = Env(**env_args)
        #     self.traj_length = traj_length

        #     self.observation_space = gym.spaces.Box(
        #         low=-100.0,
        #         high=100.0,
        #         shape=(self.env.observation_dim,),
        #     )

        #     ctrl_limits = np.array(self.env.control_limits)
        #     self.action_space = gym.spaces.Box(low=ctrl_limits[:, 0], high=ctrl_limits[:, 1])

        #     assert render_mode is None or render_mode in self.metadata["render_modes"]
        #     self.render_mode = render_mode

        #     if self.env.use_graph_capture:
        #         with wp.ScopedCapture() as capture:
        #             self.env.update()
        #         self.graph = capture.graph
        #     else:
        #         self.graph = None

        #     self._termination_buffer = wp.zeros(1, dtype=wp.bool)
        #     self._termination_buffer_cpu = wp.zeros(1, dtype=wp.bool, device="cpu", pinned=True)
        #     self._observation_buffer = wp.zeros((1, self.env.observation_dim), dtype=wp.float32)
        #     self._observation_buffer_cpu = wp.zeros(
        #         (1, self.env.observation_dim), dtype=wp.float32, device="cpu", pinned=True
        #     )
        #     self._cost_buffer = wp.zeros(1, dtype=wp.float32)
        #     self._cost_buffer_cpu = wp.zeros(1, dtype=wp.float32, device="cpu", pinned=True)

        #     self._step_count = 0

        # def _get_obs(self):
        #     self.env.compute_observations(self.env.state, self.env.control, self._observation_buffer, 0, 0)
        #     wp.copy(self._observation_buffer_cpu, self._observation_buffer)
        #     observation = self._observation_buffer_cpu.numpy()[0]
        #     return observation

        # def _get_info(self):
        #     return {}
        #     # return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

        # def reset(self, seed=None, options=None):
        #     # We need the following line to seed self.np_random
        #     super().reset(seed=seed)

        #     self.env.reset()
        #     self._step_count = 0

        #     observation = self._get_obs()
        #     info = {}

        #     if self.render_mode == "human":
        #         self._render_frame()

        #     return observation, info

        # @property
        # def has_terminated(self):
        #     wp.copy(self._termination_buffer_cpu, self._termination_buffer)
        #     return self._termination_buffer_cpu.numpy()[0]

        # def step(self, action):
        #     self.env.control_input.assign(action * self.env.control_gains)
        #     if self.env.use_graph_capture:
        #         wp.capture_launch(self.graph)
        #     else:
        #         self.env.update()

        #     # reward is single-step, not cumulative
        #     self._cost_buffer.zero_()
        #     self.env.compute_cost_termination(
        #         self.env.state,
        #         self.env.control,
        #         self._step_count,
        #         self.traj_length,
        #         self._cost_buffer,
        #         self._termination_buffer,
        #     )
        #     terminated = self._step_count >= self.traj_length or self.has_terminated
        #     wp.copy(self._cost_buffer_cpu, self._cost_buffer)
        #     reward = 10.0 - self._cost_buffer_cpu.numpy()[0]
        #     observation = self._get_obs()

        #     if self.render_mode == "human":
        #         self._render_frame()

        #     self._step_count += 1

        #     info = {}
        #     return observation, reward, terminated, False, info

        # def render(self):
        #     if self.render_mode == "rgb_array":
        #         return self._render_frame()

        # def _render_frame(self):
        #     self.env.render()

        #     if self.render_mode == "rgb_array":
        #         success = self.renderer.get_pixels(self._pixel_buffer, split_up_tiles=True, mode="rgb")
        #         assert success

        #         wp.copy(self._pixel_buffer_cpu, self._pixel_buffer)

        #         return self._pixel_buffer_cpu[0]

        # def close(self):
        #     self.env.renderer.close()

    return RslEnv(num_envs)
