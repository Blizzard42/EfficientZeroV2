# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Franka Panda environment
#
# Simulation of the Franka Emika Panda robot arm.
#
###########################################################################

import os
import math
import numpy as np
import warp as wp
import warp.sim

from warp_sim_envs.envs import Environment, run_env, IntegratorType, RenderMode


@wp.kernel
def panda_target_reaching_distance_cost(
    body_q: wp.array(dtype=wp.transform),
    ee_id: int,
    ee_offset: wp.transform,
    targets: wp.array(dtype=wp.vec3),
    env_offsets: wp.array(dtype=wp.vec3),
    num_bodies: int,
    dofs: int,
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    distance: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    ee_tf = body_q[num_bodies * env_id + ee_id]
    ee_pos = wp.transform_get_translation(ee_tf * ee_offset)

    c = float(wp.length(ee_pos - targets[env_id] - env_offsets[env_id]))

    distance[env_id] = c

    alive_bonus = 2.

    wp.atomic_add(cost, env_id, c - alive_bonus)
    
    # terminate env rollout if velocity gets too high
    if terminated:
        for i in range(dofs):
            # if abs(joint_qd[env_id * dofs + i]) > 10.0 \
            if abs(joint_qd[env_id * dofs + i]) > 10. \
                or joint_q[env_id * dofs + i] < joint_limit_lower[i] \
                or joint_q[env_id * dofs + i] > joint_limit_upper[i]:

                terminated[env_id] = True
                break

@wp.kernel
def panda_target_reaching_delta_distance_cost(
    body_q: wp.array(dtype=wp.transform),
    ee_id: int,
    ee_offset: wp.transform,
    targets: wp.array(dtype=wp.vec3),
    env_offsets: wp.array(dtype=wp.vec3),
    num_bodies: int,
    dofs: int,
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    distance: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    ee_tf = body_q[num_bodies * env_id + ee_id]
    ee_pos = wp.transform_get_translation(ee_tf * ee_offset)

    c = float(wp.length(ee_pos - targets[env_id] - env_offsets[env_id]))

    wp.atomic_add(cost, env_id, c - distance[env_id])

    distance[env_id] = c
    
    # terminate env rollout if velocity gets too high
    if terminated:
        for i in range(dofs):
            # if abs(joint_qd[env_id * dofs + i]) > 10.0 \
            if abs(joint_qd[env_id * dofs + i]) > 10.0 \
                or joint_q[env_id * dofs + i] < joint_limit_lower[i] \
                or joint_q[env_id * dofs + i] > joint_limit_upper[i]:

                terminated[env_id] = True
                break

@wp.kernel(enable_backward=False)
def resample_targets_old(
    randomize_targets: bool,
    kernel_seed: int,
    env_to_resample: wp.array(dtype=wp.bool),
    env_offsets: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    targets_render: wp.array(dtype=wp.vec3),
):
    env_id = wp.tid()
    if env_to_resample:
        if not env_to_resample[env_id]:
            return

    if randomize_targets:
        random_state = wp.rand_init(kernel_seed, env_id)
        # sample 3d polar coordinates
        theta = wp.randf(random_state, 0.0, 2.0 * wp.pi)
        phi = wp.randf(random_state, 0.0, wp.pi)
        r = 0.5
        targets[env_id] = wp.vec3(
            r * wp.sin(phi) * wp.cos(theta),
            r * wp.sin(phi) * wp.sin(theta) + 0.3,
            r * wp.cos(phi),
        )

    if targets_render:
        targets_render[env_id] = targets[env_id] + env_offsets[env_id]

@wp.kernel(enable_backward=False)
def resample_targets(
    randomize_targets: bool,
    kernel_seed: int,
    env_to_resample: wp.array(dtype=wp.bool),
    env_offsets: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    targets_render: wp.array(dtype=wp.vec3),
    target_pos_bounds: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    if env_to_resample:
        if not env_to_resample[env_id]:
            return

    if randomize_targets:
        random_state = wp.rand_init(kernel_seed, env_id)
        # sample target within target_pos_bounds
        target_x = wp.randf(random_state, 0., 1.)
        target_y = wp.randf(random_state, 0., 1.)
        target_z = wp.randf(random_state, 0., 1.)
        
        targets[env_id] = wp.vec3(
            target_x * (target_pos_bounds[0, 1] - target_pos_bounds[0, 0]) + target_pos_bounds[0, 0],
            target_y * (target_pos_bounds[1, 1] - target_pos_bounds[1, 0]) + target_pos_bounds[1, 0],
            target_z * (target_pos_bounds[2, 1] - target_pos_bounds[2, 0]) + target_pos_bounds[2, 0],
        )

    if targets_render:
        targets_render[env_id] = targets[env_id] + env_offsets[env_id]

@wp.kernel
def compute_observations_franka_panda(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    ee_index: int,
    ee_offset: wp.transform,
    targets: wp.array(dtype=wp.vec3),
    env_offsets: wp.array(dtype=wp.vec3),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    bodies_per_env: int,
    # outputs
    obs: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    for i in range(dof_q_per_env):
        obs[tid, i] = joint_q[tid * dof_q_per_env + i]
    for i in range(dof_qd_per_env):
        obs[tid, i + dof_q_per_env] = joint_qd[tid * dof_qd_per_env + i]
    for i in range(3):
        obs[tid, i + dof_q_per_env + dof_qd_per_env] = targets[tid][i]
    ee_tf = body_q[tid * bodies_per_env + ee_index]
    ee_pos = wp.transform_get_translation(ee_tf * ee_offset)
    for i in range(3):
        obs[tid, i + dof_q_per_env + dof_qd_per_env + 3] = ee_pos[i] - env_offsets[tid][i]

@wp.kernel(enable_backward=False)
def reset_init_state_franka(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    joint_q_range_min: wp.array(dtype=wp.float32),
    joint_q_range_max: wp.array(dtype=wp.float32),
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset:
        if not reset[env_id]:
            return

    if random_reset:
        random_state = wp.rand_init(seed, env_id)
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = wp.randf(
                random_state, joint_q_range_min[i], joint_q_range_max[i]
            )
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[i]
    else:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[i]

@wp.kernel(enable_backward=False)
def compute_initial_tracking_distance(
    reset: wp.array(dtype=wp.bool),
    body_q: wp.array(dtype=wp.transform),
    targets: wp.array(dtype=wp.vec3),
    ee_index: int,
    ee_offset: wp.transform,
    env_offsets: wp.array(dtype=wp.vec3),
    bodies_per_env: int,
    # outputs
    distance: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset and not reset[env_id]:
        return
    
    ee_tf = body_q[bodies_per_env * env_id + ee_index]
    ee_pos = wp.transform_get_translation(ee_tf * ee_offset)

    distance[env_id] = float(wp.length(ee_pos - targets[env_id] - env_offsets[env_id]))
    
@wp.kernel(enable_backward=False)
def apply_damping(
    joint_qd: wp.array(dtype=wp.float32),
    damping: float,
    body_mass: wp.array(dtype=wp.float32),
    # outputs
    joint_act: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # simplifying assumption: just one body per joint, fixed base
    m = body_mass[tid]
    joint_act[tid] = joint_act[tid] - damping / m * joint_qd[tid]
    # joint_act[tid] = joint_act[tid] - damping * joint_qd[tid]


class FrankaPandaEnvironment(Environment):
    robot_name = 'Franka'
    sim_name = "env_franka_panda"
    env_offset = (2.0, 0.0, 2.0)
    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_featherstone = 9
    sim_substeps_xpbd = 10

    num_rigid_contacts_per_env = 0

    activate_ground_plane = False

    integrator_type = IntegratorType.FEATHERSTONE

    show_joints = True

    show_target_point = True
    show_endeffector_point = True

    controllable_dofs = [0, 1, 2, 3, 4, 5]
    # control_gains = np.array([15.0, 10.0, 5.0, 2.0, 1.0, 1.0]) * 10.0
    control_gains = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 100.0
    control_limits = [
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]

    randomize_targets = False
    has_gripper = False

    # plot_joint_coords = True
    plot_joint_coords_qd = True
    use_graph_capture = False

    num_envs = 1

    # joint_damping = 0.5
    joint_damping = 0.

    gravity = 0.0 # disable gravity for franka

    # bounds within which to select target position for subsequent movement of gripper using RL policy ([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
    target_pos_bounds = None

    def __init__(self, 
                 seed=42, 
                 random_reset=False, 
                 random_target=True,
                 reward_type='distance', 
                 **kwargs):
        self.seed = seed
        self.random_reset = random_reset
        self.randomize_targets = random_target
        assert reward_type in ['distance', 'delta_distance']
        self.reward_type = reward_type
        super().__init__(**kwargs)
        self.distance = wp.zeros(self.num_envs, dtype=wp.float32)

        # bounds within which to select target position for subsequent movement of gripper using RL policy ([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
        self.target_pos_bounds = wp.array([[0.3, 0.7], [0.034, 0.184], [-0.35, 0.35]], dtype=wp.float32)

    def create_articulation(self, builder):
        # urdf_file = "frankaEmikaPanda.urdf" if self.has_gripper else "frankaEmikaPanda_no_gripper.urdf"
        urdf_file = (
            "frankaEmikaPanda.urdf"
            if self.has_gripper
            else "frankaEmikaPanda_simple.urdf"
        )
        path = os.path.join(
            os.path.dirname(__file__),
            f"assets/franka_description/robots/{urdf_file}",
        )
        wp.sim.parse_urdf(
            path,
            builder,
            xform=wp.transform(
                (0.0, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5),
            ),
            floating=False,
            armature=0.01,
            density=3500,
            # limit_ke=1.0e4,
            # limit_kd=1.0e2,
            limit_ke=0.,
            limit_kd=0.,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        if self.has_gripper:
            self.endeffector_id = builder.body_count - 3
            self.endeffector_offset = wp.transform([0.0, 0.0, 0.22], wp.quat_identity())
        else:
            self.endeffector_id = builder.body_count - 1
            self.endeffector_offset = wp.transform(
                [0.088, -0.22, 0.0], wp.quat_identity()
            )

        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307] 

        if self.show_endeffector_point:
            builder.add_shape_sphere(
                self.endeffector_id,
                pos=self.endeffector_offset.p,
                radius=0.025,
                density=0.0,
                has_shape_collision=False,
            )

        self.env_offsets_wp = wp.array(self.env_offsets, dtype=wp.vec3)
        self.targets = wp.array(
            np.tile([0.0, 0.0, 0.5], (self.num_envs, 1)), dtype=wp.vec3
        )
        self.targets_render = wp.empty_like(self.targets)

        for i in range(builder.joint_count):
            builder.joint_act[i] = 0.0
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_FORCE
    
    def sample_targets(self, env_ids: wp.array = None):
        wp.launch(
            resample_targets,
            dim=self.num_envs,
            inputs=[
                self.randomize_targets,
                self.seed,
                env_ids,
                self.env_offsets_wp,
                self.targets,
                self.targets_render,
                self.target_pos_bounds,
            ],
            device=self.device,
        )
        self.seed += self.num_envs

    def reset_envs(self, env_ids: wp.array = None):
        """Print distance for the envs to be reset. """
        if env_ids is not None and env_ids.numpy().any():
            # print('final distance = ', np.mean(self.distance.numpy()[env_ids.numpy()]))
            self.extras['episode'] = {}
            self.extras['episode']['final distance'] = np.mean(self.distance.numpy()[env_ids.numpy()])

        """Reset environments where env_ids buffer indicates True. Resets all envs if env_ids is None."""
        wp.launch(
            reset_init_state_franka,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed,
                self.random_reset,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_q,
                self.model.joint_qd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
            ],
            outputs=[
                self.state.joint_q,
                self.state.joint_qd,
            ],
            device=self.device,
        )
        self.seed += self.num_envs
        self.sample_targets(env_ids)

        # update maximal coordinates after reset
        if env_ids is None or env_ids.numpy().any():
            if self.uses_generalized_coordinates:
                wp.sim.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, None, self.state)
                
        
        # compute initial tracking distance
        wp.launch(
            compute_initial_tracking_distance,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.state.body_q,
                self.targets,
                self.endeffector_id,
                self.endeffector_offset,
                self.env_offsets_wp,
                self.bodies_per_env,
            ],
            outputs=[
                self.distance
            ],
            device=self.device
        )

    def before_step(self):
        wp.launch(
            apply_damping,
            dim=self.num_envs * self.dof_qd_per_env,
            inputs=[self.state.joint_qd, self.joint_damping, self.model.body_mass],
            outputs=[self.control.joint_act],
            device=self.device,
        )

    def custom_render(self, render_state, renderer):
        if self.show_target_point:
            if self.render_mode == RenderMode.OPENGL:
                renderer.render_points(
                    "targets", self.targets_render, radius=0.025, colors=(1.0, 0.0, 0.0)
                )
            elif self.render_mode == RenderMode.USD:
                renderer.render_points(
                    "targets",
                    self.targets_render.numpy(),
                    radius=0.025,
                    colors=(1.0, 0.0, 0.0),
                )

    def compute_cost_termination(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        step: int,
        max_episode_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)

        if self.reward_type == 'distance':
            wp.launch(
                panda_target_reaching_distance_cost,
                dim=self.num_envs,
                inputs=[
                    state.body_q,
                    self.endeffector_id,
                    self.endeffector_offset,
                    self.targets,
                    self.env_offsets_wp,
                    self.bodies_per_env,
                    self.dof_q_per_env,
                    state.joint_q,
                    state.joint_qd,
                    control.joint_act,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                ],
                outputs=[cost, self.distance, terminated],
                device=self.device,
            )
        elif self.reward_type == 'delta_distance':
            wp.launch(
                panda_target_reaching_delta_distance_cost,
                dim=self.num_envs,
                inputs=[
                    state.body_q,
                    self.endeffector_id,
                    self.endeffector_offset,
                    self.targets,
                    self.env_offsets_wp,
                    self.bodies_per_env,
                    self.dof_q_per_env,
                    state.joint_q,
                    state.joint_qd,
                    control.joint_act,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                ],
                outputs=[cost, self.distance, terminated],
                device=self.device,
            )

    @property
    def observation_dim(self):
        # joint q, joint qd, target, endeffector
        return self.dof_q_per_env + self.dof_qd_per_env + 3 + 3

    def compute_observations(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        if not self.uses_generalized_coordinates:
            # evaluate generalized coordinates
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)

        wp.launch(
            compute_observations_franka_panda,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                state.body_q,
                self.endeffector_id,
                self.endeffector_offset,
                self.targets,
                self.env_offsets_wp,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.bodies_per_env,
            ],
            outputs=[observations],
            device=self.device,
        )

if __name__ == "__main__":
    run_env(FrankaPandaEnvironment)
