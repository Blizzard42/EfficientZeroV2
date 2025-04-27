import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.autograd.functional

import cvxpy as cp

import gymnasium as gym
import warp as wp
import warp.sim
import warp.autograd

from warp_sim_envs.wrappers.torch import TorchEnvWrapper
from warp_sim_envs.wrappers.torch_manipulability import ManipulabilityTorchEnvWrapper
from warp_sim_envs.envs import AllegroRotateCubeEnvironment, IntegratorType
from warp_sim_envs.algos import SHAC
from warp_sim_envs.utils import function_jacobian, function_jacobian_fd, check_jacobian
from warp_sim_envs.utils.grad_utils import compare_jacobians


@wp.kernel
def assign_target_poses(
    target_orientation: wp.array(dtype=wp.quat),
    # output
    target_body_q: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()

    for i in range(4):
        target_body_q[env_id, 3+i] = target_orientation[env_id][i]

@wp.kernel
def interpolate_trajectory_smooth(
    start_poses: wp.array(dtype=wp.float32, ndim=2),
    goal_poses: wp.array(dtype=wp.float32, ndim=2),
    num_steps: int,
    # Output
    trajectories: wp.array(dtype=wp.float32, ndim=3)
):
    env_id = wp.tid()

    start_pose = start_poses[env_id]
    goal_pose = goal_poses[env_id]

    start_quat = wp.quat(start_pose[3], start_pose[4], start_pose[5], start_pose[6])
    goal_quat = wp.quat(goal_pose[3], goal_pose[4], goal_pose[5], goal_pose[6])

    num_steps_float = float(num_steps)

    for t in range(num_steps):
        # alpha = t / num_steps # Linear Interpolation
        
        alpha = 1.0 / (1.0 + wp.exp(-(float(t) - num_steps_float / 2.0) * 10.0 / num_steps_float))
        if t == 0:
            alpha = 0.0
        elif t == num_steps - 1:
            alpha == 1.0

        for i in range(3):
            trajectories[env_id, t, i] = alpha * (goal_pose[i] - start_pose[i]) + start_pose[i]

        if t == num_steps - 1:
            interpolated_quat = goal_quat
        else:
            interpolated_quat = wp.quat_slerp(start_quat, goal_quat, alpha)

        for i in range(4):
            trajectories[env_id, t, 3+i] = interpolated_quat[i]

def quat_inverse_xyzw(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize and invert quaternion(s) in [x, y, z, w] format.

    Args:
        q (torch.Tensor): Tensor of shape (..., 4) with quaternions in [x, y, z, w] format.

    Returns:
        torch.Tensor: Inverse quaternion(s), normalized, in [x, y, z, w] format.
    """
    # Normalize to unit quaternion
    q = q / q.norm(dim=-1, keepdim=True)
    
    # Extract components
    x, y, z, w = q.unbind(-1)
    
    # Conjugate and reorder as [x, y, z, w]
    return torch.stack([-x, -y, -z, w], dim=-1)

def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions in [x, y, z, w] format.

    Args:
        q1 (torch.Tensor): (..., 4) tensor of quaternions [x, y, z, w]
        q2 (torch.Tensor): (..., 4) tensor of quaternions [x, y, z, w]

    Returns:
        torch.Tensor: (..., 4) tensor of resulting quaternion [x, y, z, w]
    """
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)

    x =  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y =  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z =  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w =  w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack([x, y, z, w], dim=-1)


def init_environment(num_envs, max_episode_length, random_reset = False):
    raw_env = AllegroRotateCubeEnvironment(
        num_envs=num_envs,
        setup_renderer=True,
        requires_grad=True,
        random_reset=random_reset,
    )
    
    env = ManipulabilityTorchEnvWrapper(
        raw_env,
        render_mode="Full",
        max_episode_length=max_episode_length,
        reward_bias=0.0,
        reward_scale=10.0,
        image_width=64,
        image_height=64,
    )

    return env



position_cost_weight = 0.05
orientation_cost_weight = 1.0

cost_mask = np.ones(7)
cost_mask[:3] *= position_cost_weight
cost_mask[3:] *= orientation_cost_weight

def optimize_action_cvxpy(num_actions, current_pose, goal_pose, manipulability, max_action=0.1):
    # minimizing the cost function using cvxpy

    # print(f'{current_pose.shape=} {goal_pose.shape=} {manipulability.shape=}')

    # Set up CVX problem
    x = cp.Variable(num_actions)
    M = manipulability
    # M = get_manipulability_fd_toy(env, manip_obs_dict=env.current_obs_dict, obs_keys_manip=["object_dof_pos"]).T.cpu()
    # obj_curr = env.current_obs_dict["manip_obs"].cpu()
    obj_curr = current_pose
    obj_des = goal_pose
    obj_next = obj_curr + M @ x

    # position_cost = cp.norm((obj_next[:3] - obj_des[:3]))

    # # I asked chat gpt and it say closeness measure proxy that is convex is q1T dot q2
    # orientation_cost = cp.square(1.0 - obj_des[3:]@obj_next[3:])

    cost = cp.norm(cp.multiply(cost_mask, obj_next - obj_des))
    # cost = position_cost_weight * position_cost + orientation_cost_weight * orientation_cost

    objective = cp.Minimize(cost)
    # we also want the first 6 dofs to be 0 and the action to be small (infinity norm <= 0.1)
    # constraints = [x[:6] == 0, cp.norm(x, "inf") <= max_action]
    constraints = [cp.norm(x, "inf") <= max_action]

    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()
    cost = torch.tensor(prob.value).float()
    cost_normalized = prob.value / (torch.norm(obj_des - obj_curr) + 1e-9)
    action = torch.tensor(x.value).float()

    return cost_normalized, action

def main():
    num_envs = 1
    max_episode_length = 300

    target_object = 'cube'

    # Init environment
    env = init_environment(num_envs, max_episode_length, False)

    observation = env.reset()

    # Get trajectory
    start_poses = wp.zeros((num_envs, 7), dtype=wp.float32)
    env.env.query_target_body_q(env.state, target_object, start_poses)
    
    print(f'{env.env.target_orientations.numpy()=}')

    goal_poses = wp.clone(start_poses)
    wp.launch(
        assign_target_poses,
        dim=num_envs,
        inputs=[env.env.target_orientations],
        outputs=[goal_poses],
        device=env.device
    )

    # Trajectory
    dense_trajectories = wp.zeros((num_envs, max_episode_length, 7), dtype=wp.float32)
    wp.launch(
        interpolate_trajectory_smooth,
        dim=num_envs,
        inputs=[start_poses, goal_poses, max_episode_length],
        outputs=[dense_trajectories],
        device=env.device
    )
    dense_trajectories_torch = wp.to_torch(dense_trajectories).squeeze()

    # Optimize trajectory

    observation = env.reset()
    max_reward = -19
    min_reward = 0
    # Start with random 0 action
    observation, reward, done, extra, manipulability, current_object_pose = env.step(torch.zeros((num_envs, env.num_actions), dtype=torch.float32, device=wp.device_to_torch(env.device)))
    min_reward = min(reward, min_reward)
    max_reward = max(reward, max_reward)
    costs = []
    actions = []

    goal_index = 0
    goal_reach_deviation = 0.005


    def goal_cost(current_pose, target_pose):

        position_cost = torch.norm(current_pose[..., :3] - target_pose[..., :3])
        quat_diff = quat_multiply(current_pose[..., 3:], quat_inverse_xyzw(target_pose[..., 3:]))

        sin_half_angle = torch.sqrt(torch.sum(torch.pow(quat_diff,2)[..., :-1], axis=-1)) 
        orientation_cost = 2.0 * torch.asin(torch.minimum(sin_half_angle, torch.ones_like(sin_half_angle)))

        total_cost = orientation_cost

        return total_cost


    for i in range(max_episode_length):
        current_object_pose_cpu = current_object_pose[target_object].detach().cpu().squeeze()

        current_goal_cpu = dense_trajectories_torch[goal_index].cpu().squeeze()
        # current_goal_cpu = dense_trajectories_torch[i].cpu().squeeze()
        
        cost, action = optimize_action_cvxpy(
            env.num_actions, current_object_pose_cpu, current_goal_cpu, manipulability[target_object].cpu().squeeze(), max_action=50.0
        )
        costs.append(cost)
        actions.append(action)
        
        observation, reward, done, extra, manipulability, current_object_pose = env.step(action.to(wp.device_to_torch(env.device)))
        min_reward = min(reward, min_reward)
        max_reward = max(reward, max_reward)

        # Progress
        current_object_pose_cpu = current_object_pose[target_object].cpu().squeeze()
        if torch.norm(goal_cost(current_object_pose_cpu, current_goal_cpu)).cpu() < goal_reach_deviation:
            # Progress 1 frame
            goal_index += 1
        else:
            # Set to nearest to current position
            cost_to_trajectory = goal_cost(current_object_pose[target_object], dense_trajectories_torch)
            # print(f'{cost_to_trajectory.shape=} {current_object_pose[target_object].shape=} {dense_trajectories_torch.shape=}')
            new_goal_index = torch.argmin(
                cost_to_trajectory
            )
            # print(f'{goal_index=} {new_goal_index=}')
            goal_index = new_goal_index
            # print(f'{goal_costs.shape=} {current_object_pose.shape=} {dense_trajectories_torch=}')
        
        if goal_index >= max_episode_length:
            # Reach final goal
            print(f'Reached final goal at {i} iteration {goal_index=}')
            break

        if done:
            print(f'Reset at {i} iteration')
            break

    print(f'{costs=}')
    print(f'{actions=}')
    print("MIN REWARD:", min_reward)
    print("MAX REWARD:", max_reward)

    fig = plt.figure()
    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()