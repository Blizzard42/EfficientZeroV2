import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import warp as wp

# Import the necessary classes from your WARP setup
from .warp_sim_envs.envs.env_allegro_rotate_cube import AllegroRotateCubeEnvironment
from .warp_sim_envs.wrappers.torch_manipulability import ManipulabilityTorchEnvWrapper # Adjust import paths if needed

# Default max steps if not provided in config
DEFAULT_MAX_STEPS = 300

class HandEnvGymWrapper(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30} # Example

    def __init__(self, seed=None, config=None, **kwargs):
        super().__init__()

        # Ensure WARP is initialized in this process
        # if not wp.device():
        #     wp.init()

        self.config = config if config is not None else {}
        self.max_episode_steps = self.config.get('max_episode_steps', DEFAULT_MAX_STEPS)
        self._render_mode = self.config.get('render_mode', None) # Or get from kwargs
        # --- Instantiate the underlying WARP env and Torch wrapper ---
        # Assuming num_envs is always 1 for standard Gym interface
        # Pass requires_grad=False if EZV2 doesn't need gradients
        # NOTE: device handling might be needed if run on specific GPUs via Ray
        self.warp_torch_env = ManipulabilityTorchEnvWrapper(
            AllegroRotateCubeEnvironment(
                num_envs=1,
                seed=seed,
                setup_renderer=(self._render_mode is not None and self._render_mode.lower() != 'none'),
                requires_grad=True,
                random_reset=True,
            ),
            render_mode=self._render_mode,
            max_episode_length=self.max_episode_steps,
            requires_grad=True,
            reward_bias=0.0,
            reward_scale=10.0,
            image_width=64,
            image_height=64,
        )

        # Initialize self.device
        if isinstance(self.warp_torch_env.device, torch.device):
            self.device = self.warp_torch_env.device
        elif isinstance(self.warp_torch_env.device, str):
            self.device = torch.device(self.warp_torch_env.device) # Convert string to torch.device
        elif isinstance(self.warp_torch_env.device, int): # Handle case if it's just a GPU index
            self.device = torch.device(f"cuda:{self.warp_torch_env.device}")
        else:
            # Default or raise error if the device isn't recognized
            print(f"Warning: Unrecognized device '{self.warp_torch_env.device}' from warp_torch_env. Defaulting to CPU.")
            self.device = torch.device("cpu")
            # Or raise TypeError(f"Invalid device identifier from warp_torch_env: {warp_device_identifier}")

        # --- Define Observation Space ---
        # From analysis: 55 dimensions
        self.observation_dim = 55
        low_obs = -np.inf * np.ones(self.observation_dim, dtype=np.float32)
        high_obs = np.inf * np.ones(self.observation_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)

        # --- Define Action Space ---
        # From analysis: 19 dimensions (Allegro hand + base)
        # Assuming normalized actions [-1, 1]
        self.action_dim = 19
        low_act = -1.0 * np.ones(self.action_dim, dtype=np.float32)
        high_act = 1.0 * np.ones(self.action_dim, dtype=np.float32)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        self._current_step = 0
        self.seed(seed) # Ensure seeding propagates if needed

    def seed(self, seed=None):
        # Basic seeding, might need refinement depending on warp_torch_env's seeding
        super().reset(seed=seed)
        # Potentially call a seeding method on self.warp_torch_env if it exists

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handle seeding

        # Reset the underlying env (using force_reset for full reset)
        # Pass grads=False as we don't need gradients for standard RL reset
        obs_torch = self.warp_torch_env.reset(grads=False, force_reset=False)

        # Convert observation to NumPy
        observation = obs_torch.detach().cpu().numpy().squeeze() # Squeeze if num_envs=1

        # Reset step counter
        self._current_step = 0

        return observation

    def step(self, action):
        if not isinstance(action, np.ndarray):
            raise ValueError("Action must be a NumPy array")
        if action.shape != self.action_space.shape:
             raise ValueError(f"Action shape mismatch: expected {self.action_space.shape}, got {action.shape}")

        # Convert NumPy action to Torch tensor, add batch dim (num_envs=1)
        action_torch = torch.from_numpy(action).float().unsqueeze(0).to(self.device)

        # Step the underlying environment
        # Ignore next_q, next_qd, manipulability, object_poses
        obs_torch, reward_torch, done_torch, extras, manipulability, _ = self.warp_torch_env.step(action_torch)

        # Convert results back to NumPy / Python scalars
        observation = obs_torch.detach().cpu().numpy().squeeze()
        reward = reward_torch.item()
        if reward > 0.87 or reward < -31:
            print(f"WARNING: Reward of {reward} outside expected range of [-31, 0.87]")

        terminated = done_torch.item() # Assuming done means terminated

        # Handle truncation (check if TimeLimit info is in extras, or use step count)
        self._current_step += 1
        truncated = self._current_step >= self.max_episode_steps
        # Check if TimeLimit added info (unlikely through this wrapper chain)
        # truncated = truncated or extras.get("TimeLimit.truncated", False)

        # Populate info dict (optional)
        info = {'raw_reward': reward}
        # You could add things from 'extras' here if needed, e.g.:
        # info['primal_reward'] = extras.get('primal', reward).item()
        return observation, reward, terminated or truncated, info

    def render(self, mode=None):
        # Delegate rendering to the underlying environment
        self.warp_torch_env.render()

    def close(self):
        # Delegate closing to the underlying environment
        self.warp_torch_env.close()