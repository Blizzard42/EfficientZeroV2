# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Humanoid environment
#
# Shows how to set up a simulation of a rigid-body Humanoid articulation based
# on the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import numpy as np
import warp as wp
import warp.sim
import warp.examples

from warp_sim_envs.envs import Environment, run_env, IntegratorType

ZERO_GRAVITY = False


class HumanoidEnvironment(Environment):
    sim_name = "env_humanoid"
    env_offset = (2.0, 0.0, 2.0)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 16
    sim_substeps_featherstone = 6
    sim_substeps_xpbd = 6

    # render_mode = RenderMode.NONE

    # integrator_type = IntegratorType.FEATHERSTONE
    # integrator_type = IntegratorType.EULER

    xpbd_settings = dict(
        iterations=2,
        # joint_linear_relaxation=0.7,
        # joint_angular_relaxation=0.5,
        # rigid_contact_relaxation=1.0,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.4,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    use_tiled_rendering = False
    show_joints = False

    # num_envs = 4096
    num_envs = 100

    control_dim = 21
    control_gains = (
        np.array(
            [
                200,
                200,
                200,
                200,
                200,
                600,
                400,
                100,
                100,
                200,
                200,
                600,
                400,
                100,
                100,
                100,
                100,
                200,
                100,
                100,
                200,
            ]
        )
        * 0.2
    )
    # control_gains = np.array([0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 200, 200, 600, 400, 100, 100, 100, 100, 200, 100, 100, 200]) * 0.05
    control_limits = [(-1.0, 1.0)] * control_dim
    controllable_dofs = list(range(control_dim))

    activate_ground_plane = not ZERO_GRAVITY

    def create_articulation(self, builder):
        wp.sim.parse_mjcf(
            os.path.join(warp.examples.get_asset_directory(), "nv_humanoid.xml"),
            builder,
            stiffness=0.0,
            damping=0.1,
            armature=0.007,
            armature_scale=10.0,
            contact_ke=2.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e4,
            contact_mu=0.5,
            contact_restitution=0.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            up_axis="y",
            collapse_fixed_joints=True,
        )

        for i in range(builder.joint_axis_count):
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_FORCE

        # # find the names of the control inputs
        # joint_id = 1
        # joint_type_names = {
        #     wp.sim.JOINT_BALL: "ball",
        #     wp.sim.JOINT_REVOLUTE: "hinge",
        #     wp.sim.JOINT_PRISMATIC: "slide",
        #     wp.sim.JOINT_UNIVERSAL: "universal",
        #     wp.sim.JOINT_COMPOUND: "compound",
        #     wp.sim.JOINT_FREE: "free",
        #     wp.sim.JOINT_FIXED: "fixed",
        #     wp.sim.JOINT_DISTANCE: "distance",
        #     wp.sim.JOINT_D6: "D6",
        # }
        # joint_type = builder.joint_type
        # while (
        #     joint_id < len(joint_type) - 1
        #     and joint_type[joint_id] == wp.sim.JOINT_FIXED
        # ):
        #     # skip fixed joints
        #     joint_id += 1
        # q_start = builder.joint_q_start
        # qd_start = builder.joint_qd_start
        # axis_start = builder.joint_axis_start
        # qd_i = qd_start[joint_id]
        # axis_i = axis_start[joint_id]
        # for dim in range(builder.joint_axis_count):
        #     joint_name = joint_type_names[joint_type[joint_id]]
        #     if (
        #         joint_id < builder.joint_count - 1
        #         and q_start[joint_id + 1] == dim + 1
        #     ):
        #         joint_id += 1
        #         qd_i = qd_start[joint_id]
        #         axis_i = axis_start[joint_id]
        #     else:
        #         qd_i += 1
        #     name = f"$\\mathbf{{q_{{{dim}}}}}$ ({builder.joint_name[joint_id]} / {joint_name} {joint_id})"
        #     print(name)

        if ZERO_GRAVITY:
            builder.gravity = 0.0

        self.start_rot = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -wp.pi * 0.5)
        self.inv_start_rot_wp = wp.quat_inverse(self.start_rot)
        builder.joint_q[:7] = [
            0.0,
            1.35,
            0.0,
            *self.start_rot,
        ]

        self.basis_vec0_wp = wp.vec3(1.0, 0.0, 0.0)
        self.basis_vec1_wp = wp.vec3(0.0, 1.0, 0.0)
        self.heading_vec = wp.vec3(1.0, 0.0, 0.0)
        self.up_vec = wp.vec3(0.0, 1.0, 0.0)
        self.target_pos = wp.vec3(1000.0, 0.0, 0.0)

        self.termination_height = 0.74
        # self.termination_height = 0.54

        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.device)

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
            calculate_cost,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                state.body_qd,
                self.env_offsets_wp,
                self.inv_start_rot_wp,
                self.target_pos,
                self.basis_vec0_wp,
                self.basis_vec1_wp,
                self.bodies_per_env,
                self.termination_height,
            ],
            outputs=[cost, terminated],
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
            calculate_observations,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                state.body_qd,
                state.joint_q,
                state.joint_qd,
                self.env_offsets_wp,
                self.inv_start_rot_wp,
                self.target_pos,
                self.basis_vec0_wp,
                self.basis_vec1_wp,
                self.bodies_per_env,
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[observations],
            device=self.device,
        )


@wp.kernel
def calculate_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    start_pos: wp.array(dtype=wp.vec3),
    inv_start_rot: wp.quat,
    target_pos: wp.vec3,
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    num_bodies: int,
    termination_height: float,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    tid = wp.tid()

    st_pos = start_pos[tid]
    body_id = num_bodies * tid

    torso_pos = wp.transform_get_translation(body_q[body_id])
    torso_rot = wp.transform_get_rotation(body_q[body_id])

    lin_vel = wp.spatial_bottom(body_qd[body_id])
    # ang_vel = wp.spatial_top(body_qd[body_id])

    to_target = target_pos + st_pos - torso_pos

    target_dir = wp.normalize(to_target)
    torso_quat = torso_rot * inv_start_rot

    up_vec = wp.quat_rotate(torso_quat, basis_vec1)
    heading_vec = wp.quat_rotate(torso_quat, basis_vec0)

    torso_target_proj = wp.dot(target_dir, heading_vec)

    up_reward = 0.1 * up_vec[1]
    heading_reward = torso_target_proj
    height_delta = torso_pos[1] - termination_height
    height_reward = (
        0.5 * height_delta + 2.0
    )  # 2.5 #obs_buf[tid * num_obs] - termination_height + 0.5
    progress_reward = lin_vel[0]

    total = progress_reward + up_reward + heading_reward + height_reward
    # + torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

    cost[tid] = -total

    # reset agents that fall below the termination height
    if height_delta < 0.0:
        terminated[tid] = True


@wp.kernel
def calculate_observations(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    # actions: wp.array(dtype=float),
    start_pos: wp.array(dtype=wp.vec3),
    inv_start_rot: wp.quat,
    target_pos: wp.vec3,  # todo - array
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    num_bodies: int,
    num_q: int,
    num_qd: int,
    # outputs
    obs_buf: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()

    st_pos = start_pos[tid]

    body_id = num_bodies * tid

    torso_pos = wp.transform_get_translation(body_q[body_id])
    torso_rot = wp.transform_get_rotation(body_q[body_id])

    lin_vel = wp.spatial_bottom(body_qd[body_id])
    ang_vel = wp.spatial_top(body_qd[body_id])

    to_target = target_pos + st_pos - torso_pos

    target_dir = wp.normalize(to_target)
    torso_quat = torso_rot * inv_start_rot

    up_vec = wp.quat_rotate(torso_quat, basis_vec1)
    heading_vec = wp.quat_rotate(torso_quat, basis_vec0)

    torso_target_proj = wp.dot(target_dir, heading_vec)

    obs_idx = 0
    obs_buf[tid][obs_idx] = torso_pos[1]
    obs_idx = obs_idx + 1

    for i in range(4):
        obs_buf[tid][obs_idx + i] = torso_rot[i]

    obs_idx = obs_idx + 4

    for i in range(3):
        obs_buf[tid][obs_idx + i] = lin_vel[i]

    obs_idx = obs_idx + 3

    for i in range(3):
        obs_buf[tid][obs_idx + i] = ang_vel[i]

    obs_idx = obs_idx + 3

    # 11

    # TODO pass as arguments
    num_root_q = 7
    num_root_qd = 6

    num_joints = num_q - num_root_q
    # num_act = num_joints

    q_offset = num_q * tid
    qd_offset = num_qd * tid

    # act_offset = (num_act + 6) * tid

    for i in range(num_joints):
        obs_buf[tid][obs_idx + i] = joint_q[q_offset + num_root_q + i]

    obs_idx = obs_idx + num_joints

    qd_scale = 0.1
    for i in range(num_joints):
        obs_buf[tid][obs_idx + i] = qd_scale * joint_qd[qd_offset + num_root_qd + i]

    obs_idx = obs_idx + num_joints

    # 11 + 42 = 53

    obs_buf[tid][obs_idx] = up_vec[1]
    obs_idx = obs_idx + 1

    obs_buf[tid][obs_idx] = torso_target_proj
    obs_idx = obs_idx + 1

    # 55

    # if add_actions != 0:
    #     for i in range(0, num_act):
    #         obs_buf[tid][obs_idx + i] = actions[act_offset + i]

    #     obs_idx = obs_idx + num_act

    # 76


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


def apply_sinusoidal_control(self: HumanoidEnvironment):
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
            self.control_input,
            self.sim_time_wp,
        ],
        device=self.device,
    )


if __name__ == "__main__":
    # HumanoidEnvironment.before_step = apply_sinusoidal_control
    run_env(HumanoidEnvironment)
