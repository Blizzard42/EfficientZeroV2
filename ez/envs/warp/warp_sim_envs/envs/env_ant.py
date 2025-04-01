# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Ant environment
#
# Shows how to set up a simulation of a rigid-body Ant articulation based on
# the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import warp as wp

# wp.set_module_options({"max_unroll": 1})
import warp.sim
import numpy as np

from warp_sim_envs.envs import Environment, run_env, IntegratorType

ZERO_GRAVITY = False


@wp.kernel
def ant_running_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    num_bodies: int,
    dofs: int,
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    base_q = body_q[num_bodies * env_id + 0]
    base_pos = wp.transform_get_translation(base_q)

    base_vel = wp.spatial_bottom(body_qd[num_bodies * env_id + 0])
    up = wp.vec3(0.0, 1.0, 0.0)

    # only move forward, not sideways
    c = -base_vel[0] * 100.0 + base_vel[1] ** 2.0 + base_vel[2] ** 2.0

    # point upwards
    # c -= 1.0 - wp.dot(wp.transform_vector(base_q, up), up)

    # penalize force and velocity
    # for i in range(dofs):
    #     c += joint_act[env_id * dofs + i] ** 2.0 * 0.01
    #     c += joint_qd[env_id * dofs + i] ** 2.0 * 0.001

    wp.atomic_add(cost, env_id, c)

    # terminate env rollout if base falls below certain height
    if terminated:
        if base_pos[1] < 0.3:
            terminated[env_id] = True


def ant_running_cost_dflex(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    inv_start_rot: wp.quat,
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    dof_q: int,
    dof_qd: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()
    
    torso_pos = wp.vec3(joint_q[dof_q * env_id + 0],
                        joint_q[dof_q * env_id + 1],
                        joint_q[dof_q * env_id + 2])
    torso_quat = wp.quat(joint_q[dof_q * env_id + 3],
                        joint_q[dof_q * env_id + 4],
                        joint_q[dof_q * env_id + 5],
                        joint_q[dof_q * env_id + 6])
    lin_vel = wp.vec3(joint_qd[dof_qd * env_id + 3],
                      joint_qd[dof_qd * env_id + 4],
                      joint_qd[dof_qd * env_id + 5])
    ang_vel = wp.vec3(joint_qd[dof_qd * env_id + 0],
                      joint_qd[dof_qd * env_id + 1],
                      joint_qd[dof_qd * env_id + 2])
    
    # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
    lin_vel = lin_vel - wp.cross(torso_pos, ang_vel)

    up_vec = wp.quat_rotate(torso_quat, basis_vec1)
    heading_vec = wp.quat_rotate(torso_quat, basis_vec0)

    up_reward = up_vec[1] * 0.1
    heading_reward = heading_vec[0]
    # height_reward = torso_pos[1] - 0.3
    progress_reward = lin_vel[0]

    c = -progress_reward - up_reward - heading_reward #- height_reward
    # c = -lin_vel[0] #* 5. + lin_vel[1] ** 2. + lin_vel[2] ** 2.

    wp.atomic_add(cost, env_id, c)

    if terminated:
        if torso_pos[1] < 0.3:
            terminated[env_id] = True


@wp.kernel(enable_backward=False)
def sinusoidal_control(
    num_dofs: int,
    control_gains: wp.array(dtype=wp.float32),
    control_limits: wp.array(dtype=wp.float32, ndim=2),
    dt: float,
    # outputs
    joint_act: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()

    time = sim_time[0] + dt

    for i in range(num_dofs):
        joint_act[env_id * num_dofs + i] = (
            wp.sin(time * 0.2 + wp.float32(i) * 0.5) * 0.5 * control_gains[i]
        )

    sim_time[0] = time


@wp.kernel
def compute_observations_ant_simple(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    # outputs
    obs: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    obs[tid, 0] = joint_q[tid * dof_q_per_env + 1]
    for i in range(3, dof_q_per_env):
        obs[tid, i - 2] = joint_q[tid * dof_q_per_env + i]
    for i in range(dof_qd_per_env):
        obs[tid, i + dof_q_per_env - 2] = joint_qd[tid * dof_qd_per_env + i]


@wp.kernel
def compute_observations_ant_dflex(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    inv_start_rot: wp.quat,
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    dof_q: int,
    dof_qd: int,
    # outputs
    obs: wp.array(dtype=float, ndim=2),
):
    env_id = wp.tid()

    torso_pos = wp.vec3(
        joint_q[dof_q * env_id + 0],
        joint_q[dof_q * env_id + 1],
        joint_q[dof_q * env_id + 2],
    )
    torso_quat = wp.quat(
        joint_q[dof_q * env_id + 3],
        joint_q[dof_q * env_id + 4],
        joint_q[dof_q * env_id + 5],
        joint_q[dof_q * env_id + 6],
    )
    lin_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 3],
        joint_qd[dof_qd * env_id + 4],
        joint_qd[dof_qd * env_id + 5],
    )
    ang_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 0],
        joint_qd[dof_qd * env_id + 1],
        joint_qd[dof_qd * env_id + 2],
    )

    # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
    lin_vel = lin_vel - wp.cross(torso_pos, ang_vel)

    up_vec = wp.quat_rotate(torso_quat, basis_vec1)
    heading_vec = wp.quat_rotate(torso_quat, basis_vec0)

    obs[env_id, 0] = torso_pos[1]  # 0
    for i in range(4):  # 1:5
        obs[env_id, 1 + i] = torso_quat[i]
    for i in range(3):  # 5:8
        obs[env_id, 5 + i] = lin_vel[i]
    for i in range(3):  # 8:11
        obs[env_id, 8 + i] = ang_vel[i]
    for i in range(8):  # 11:19
        obs[env_id, 11 + i] = joint_q[dof_q * env_id + 7 + i]
    for i in range(8):  # 19:27
        obs[env_id, 19 + i] = joint_qd[dof_qd * env_id + 6 + i]
    obs[env_id, 27] = up_vec[1]  # 27
    obs[env_id, 28] = heading_vec[0]  # 28


@wp.kernel(enable_backward=False)
def reset_ant(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset:
        if not reset[env_id]:
            return

    for i in range(dof_q_per_env):
        joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[
            env_id * dof_q_per_env + i
        ]
    for i in range(dof_qd_per_env):
        joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[
            env_id * dof_q_per_env + i
        ]

    if random_reset:
        random_state = wp.rand_init(seed, env_id)

        # randomize base position
        base_position_perturbation = wp.vec3(0.2, 0.1, 0.2)
        for i in range(3):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[
                env_id * dof_q_per_env + i
            ] + base_position_perturbation[i] * wp.randf(random_state, -1.0, 1.0)

        # randomize base orientation
        angle = wp.randf(random_state, -1.0, 1.0) * wp.pi / 6.0
        axis = wp.vec3(
            wp.randf(random_state, -1.0, 1.0),
            wp.randf(random_state, -1.0, 1.0),
            wp.randf(random_state, -1.0, 1.0),
        )
        axis = wp.normalize(axis)
        default_quat = wp.quat(
            default_joint_q_init[3],
            default_joint_q_init[4],
            default_joint_q_init[5],
            default_joint_q_init[6],
        )
        delta_quat = wp.quat_from_axis_angle(axis, angle)
        quat = default_quat * delta_quat
        for i in range(4):
            joint_q[env_id * dof_q_per_env + i + 3] = quat[i]

        # randomize joint angles
        for i in range(8):
            joint_q[env_id * dof_q_per_env + i + 7] = default_joint_q_init[
                env_id * dof_q_per_env + 7 + i
            ] + wp.randf(random_state, -0.2, 0.2)

        # randomize base angular velocities
        for i in range(3):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[
                env_id * dof_qd_per_env + i
            ] + 0.25 * wp.randf(random_state, -1.0, 1.0)

        # randomize base position velocities
        for i in range(3, 6):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[
                env_id * dof_qd_per_env + i
            ] + 0.1 * wp.randf(random_state, -1.0, 1.0)

        # randomize joint velocities
        for i in range(7, dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[
                env_id * dof_qd_per_env + i
            ] + 0.25 * wp.randf(random_state, -1.0, 1.0)

    # if random_reset:
    #     random_state = wp.rand_init(seed, env_id)
    #     for i in range(7, dof_q_per_env):
    #         joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[i] + wp.randf(random_state, -0.2, 0.2)
    #     for i in range(dof_qd_per_env):
    #         joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[i] + wp.randf(random_state, -0.25, 0.25)


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


class AntEnvironment(Environment):
    robot_name = "Ant"
    sim_name = "env_ant"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=0.5)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_featherstone = 16
    sim_substeps_xpbd = 8

    # xpbd_settings = dict(iterations=2)
    xpbd_settings = dict(iterations=1)

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 100.0

    # use_graph_capture = False

    # handle_collisions_once_per_step = True

    integrator_type = IntegratorType.FEATHERSTONE
    # integrator_type = IntegratorType.EULER
    integrator_type = IntegratorType.XPBD

    # plot_joint_coords = True
    # use_graph_capture = False

    # num_envs = 4096
    # num_envs = 1

    separate_ground_contacts = integrator_type != IntegratorType.XPBD
    activate_ground_plane = not ZERO_GRAVITY

    controllable_dofs = np.arange(8)
    control_gains = np.ones(8) * 500.0
    control_limits = [(-1.0, 1.0)] * 8

    joint_vel_obs_scaling = 0.1
    up_rew_scale = 0.1
    termination_height = 0.27
    action_penalty = 0.0

    def __init__(self, seed=42, random_reset=True, **kwargs):
        self.seed = seed
        self.random_reset = random_reset
        super().__init__(**kwargs)
        self.after_init()

    def create_articulation(self, builder):
        contact_ke = 2e4
        contact_kd = 1e2
        contact_kf = 1e4
        contact_mu = 0.75
        wp.sim.parse_mjcf(
            os.path.join(
                os.path.dirname(wp.sim.__file__), "../examples/assets/nv_ant.xml"
            ),
            builder,
            stiffness=0.0,
            damping=1.0,
            armature=0.1,
            contact_ke=contact_ke,
            contact_kd=contact_kd,
            contact_kf=contact_kf,
            contact_mu=contact_mu,
            limit_ke=1.0e3,
            limit_kd=1.0e2,
            enable_self_collisions=False,
            up_axis="y",
            collapse_fixed_joints=True,
        )
        builder.set_ground_plane(
            ke=contact_ke,
            kd=contact_kd,
            kf=contact_kf,
            mu=contact_mu,
        )

        self.start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
        self.inv_start_rot = wp.quat_inverse(self.start_rot)

        builder.joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
        builder.joint_q[:7] = [
            0.0,
            1.0,
            0.0,
            *self.start_rot,
        ]

        for i in range(builder.joint_axis_count):
            builder.joint_act[i] = 0.0
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_FORCE
            # builder.joint_q[i + 7] = 0.5 * (builder.joint_limit_lower[i] + builder.joint_limit_upper[i])

        # print(builder.joint_q)
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.device)

        if ZERO_GRAVITY:
            builder.gravity = 0.0

        print(builder.joint_q)
        # result = builder.sort_joints()
        # result = builder.sort_joints(list(range(builder.joint_count)))
        # print(result)
        print(builder.joint_q)

    def after_init(self):
        # create additional variables
        self.start_torso_pos = wp.array(self.model.joint_q.numpy().reshape(self.num_envs, -1)[:, 0:3].reshape(-1).copy())
        
        self.start_rot = wp.quat(self.model.joint_q.numpy()[3:7])
        self.inv_start_rot = wp.quat_inverse(self.start_rot)

        self.basis_vec0 = wp.vec3(1., 0., 0.)
        self.basis_vec1 = wp.vec3(0., 0., 1.)

    def reset_envs(self, env_ids: wp.array = None):
        """Print distance for the envs to be reset. """
        if env_ids is not None and env_ids.numpy().any():
            self.extras['episode'] = {}
            
        """Reset environments where env_ids buffer indicates True. Resets all envs if env_ids is None."""
        wp.launch(
            reset_ant,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed,
                self.random_reset,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_q,
                self.model.joint_qd,
            ],
            outputs=[
                self.state.joint_q,
                self.state.joint_qd,
            ],
            device=self.device,
        )
        self.seed += self.num_envs

        # update maximal coordinates after reset
        if env_ids is None or env_ids.numpy().any():
            if self.uses_generalized_coordinates:
                wp.sim.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, None, self.state)
                
    # def before_step(self):
    #     wp.launch(
    #         apply_damping,
    #         dim=self.num_envs * self.dof_qd_per_env,
    #         inputs=[self.state.joint_qd, self.joint_damping, self.model.body_mass],
    #         outputs=[self.control.joint_act],
    #         device=self.device,
    #     )

    def compute_cost_termination(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)

        wp.launch(
            ant_running_cost,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                state.body_qd,
                self.bodies_per_env,
                self.dof_q_per_env,
                state.joint_qd,
                control.joint_act,
            ],
            outputs=[cost, terminated],
            device=self.device,
        )

        wp.launch(
            ant_running_cost,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.inv_start_rot,
                self.basis_vec0,
                self.basis_vec1,
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[cost, terminated],
            device=self.device,
        )


def apply_sinusoidal_control(self: AntEnvironment, control, **_):
    wp.launch(
        sinusoidal_control,
        dim=self.num_envs,
        inputs=[
            self.control_dim,
            self.control_gains_wp,
            self.control_limits_wp,
            self.sim_dt,
        ],
        outputs=[
            control.joint_act,
            self.sim_time_wp,
        ],
        device=self.device,
    )


if __name__ == "__main__":
    AntEnvironment.before_step = apply_sinusoidal_control
    run_env(AntEnvironment)
