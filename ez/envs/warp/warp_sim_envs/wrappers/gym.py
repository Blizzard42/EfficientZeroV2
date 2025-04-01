import gymnasium as gym

import numpy as np
import warp as wp

from warp_sim_envs.envs import Environment, RenderMode
from .utils import get_observation_space, get_action_space


def register_gym_env(
    env_name, Env, traj_length=500, env_args={}, registration_kwargs={}
):
    class GymEnv(gym.Env):
        metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

        def __init__(self, render_mode=None):
            env_render_mode = RenderMode.NONE
            if render_mode == "human" or render_mode == "rgb_array":
                env_render_mode = RenderMode.OPENGL
            if render_mode == "rgb_array":
                env_args["use_tiled_rendering"] = True
                Env.opengl_render_settings = dict(scaling=3.0, draw_axis=False)
                Env.opengl_tile_render_settings = dict(tile_width=128, tile_height=128)
            env_args["render_mode"] = env_render_mode
            env_args["num_envs"] = 1
            self.env: Environment = Env(**env_args)
            self.traj_length = traj_length
            self.render_mode = render_mode

            if render_mode == "rgb_array":
                tile_height = self.env.opengl_tile_render_settings["tile_height"]
                tile_width = self.env.opengl_tile_render_settings["tile_width"]
                shape = (tile_height, tile_width, 3)
                self._pixel_buffer = wp.empty(
                    (1, *shape), dtype=wp.uint8, device=self.env.device
                )
                self._pixel_buffer_cpu = wp.empty(
                    (1, *shape), dtype=wp.uint8, device="cpu", pinned=True
                )

            self.observation_space = get_observation_space(self.env, render_mode)
            self.action_space = get_action_space(self.env)

            assert render_mode is None or render_mode in self.metadata["render_modes"]
            self.render_mode = render_mode

            if self.env.use_graph_capture:
                with wp.ScopedCapture() as capture:
                    self.env.update()
                self.graph = capture.graph
            else:
                self.graph = None

            self._termination_buffer = wp.zeros(1, dtype=wp.bool)
            self._termination_buffer_cpu = wp.zeros(
                1, dtype=wp.bool, device="cpu", pinned=True
            )
            self._observation_buffer = wp.zeros(
                (1, self.env.observation_dim), dtype=wp.float32
            )
            self._observation_buffer_cpu = wp.zeros(
                (1, self.env.observation_dim),
                dtype=wp.float32,
                device="cpu",
                pinned=True,
            )
            self._cost_buffer = wp.zeros(1, dtype=wp.float32)
            self._cost_buffer_cpu = wp.zeros(
                1, dtype=wp.float32, device="cpu", pinned=True
            )

            self._step_count = 0

        def _get_obs(self):
            if self.render_mode == "rgb_array":
                return self._render_frame()
            self.env.compute_observations(
                self.env.state, self.env.control, self._observation_buffer, 0, 0
            )
            wp.copy(self._observation_buffer_cpu, self._observation_buffer)
            observation = self._observation_buffer_cpu.numpy()[0]
            return observation

        def _get_info(self):
            return {}
            # return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

        def reset(self, seed=None, options=None):
            # We need the following line to seed self.np_random
            super().reset(seed=seed)

            self.env.reset()
            self._step_count = 0

            observation = self._get_obs()
            info = {}

            if self.render_mode == "human":
                self._render_frame()

            return observation, info

        @property
        def has_terminated(self):
            wp.copy(self._termination_buffer_cpu, self._termination_buffer)
            return self._termination_buffer_cpu.numpy()[0]

        def step(self, action):
            self.env.control_input.assign(action * self.env.control_gains)
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
                self._termination_buffer,
            )
            terminated = self._step_count >= self.traj_length or self.has_terminated
            wp.copy(self._cost_buffer_cpu, self._cost_buffer)
            reward = 10.0 - self._cost_buffer_cpu.numpy()[0]
            observation = self._get_obs()

            if self.render_mode == "human":
                self._render_frame()

            self._step_count += 1

            info = {}
            return observation, reward, terminated, False, info

        def render(self):
            if self.render_mode == "rgb_array":
                return self._render_frame()

        def _render_frame(self):
            self.env.render()

            if self.render_mode == "rgb_array":
                success = self.env.renderer.get_pixels(
                    self._pixel_buffer, split_up_tiles=True, mode="rgb", use_uint8=True
                )
                assert success

                wp.copy(self._pixel_buffer_cpu, self._pixel_buffer)

                return self._pixel_buffer_cpu.numpy()[0]

        def close(self):
            if self.env is not None and self.env.renderer is not None:
                self.env.renderer.close()

    gym.envs.registration.register(
        id=env_name, entry_point=GymEnv, **registration_kwargs
    )

    return GymEnv


from warp_sim_envs.envs.env_cartpole import CartpoleEnvironment

CartpoleEnv = register_gym_env("WarpSimCartPole-v1", CartpoleEnvironment, traj_length=100)
