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
import warp.sim
import numpy as np

from warp_sim_envs.envs import AntEnvironment, Environment, run_env, IntegratorType

# wp.config.verify_cuda = True

ZERO_GRAVITY = False


@wp.kernel
def anymal_running_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    env_offsets: wp.array(dtype=wp.vec3),
    num_bodies: int,
    dofs: int,
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()
    env_pos = env_offsets[env_id]

    base_q = body_q[num_bodies * env_id + 0]
    base_pos = wp.transform_get_translation(base_q) - env_pos

    base_vel = wp.spatial_bottom(body_qd[num_bodies * env_id + 0])
    up = wp.vec3(0.0, 1.0, 0.0)

    # only move forward, not sideways
    c = -base_vel[0] * 100.0 + 0.1 * base_vel[1] ** 2.0 + base_vel[2] ** 2.0

    # point upwards
    c += wp.dot(wp.transform_vector(base_q, up), up) ** 2.0

    # penalize force and velocity
    # for i in range(dofs):
    #     c += joint_act[env_id * dofs + i] ** 2.0 * 0.01
    #     c += joint_qd[env_id * dofs + i] ** 2.0 * 0.001

    wp.atomic_add(cost, env_id, c)

    # terminate env rollout if base falls below certain height
    if terminated:
        if base_pos[1] < 0.1:
            terminated[env_id] = True


class AnymalEnvironment(AntEnvironment):
    sim_name = "env_anymal"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=0.5)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_featherstone = 10
    sim_substeps_xpbd = 8

    xpbd_settings = dict(iterations=2)

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    # use_graph_capture = False

    # integrator_type = IntegratorType.EULER
    # integrator_type = IntegratorType.FEATHERSTONE
    integrator_type = IntegratorType.XPBD

    # XXX separate handling of ground contacts
    separate_ground_contacts = integrator_type != IntegratorType.XPBD

    load_visuals = True

    # plot_joint_coords = True
    use_graph_capture = False

    # num_envs = 4096
    num_envs = 1

    activate_ground_plane = not ZERO_GRAVITY

    action_strength = 150.0

    controllable_dofs = np.arange(12)
    control_gains = (
        np.array(
            [
                50.0,  # LF_HAA
                40.0,  # LF_HFE
                8.0,  # LF_KFE
                50.0,  # RF_HAA
                40.0,  # RF_HFE
                8.0,  # RF_KFE
                50.0,  # LH_HAA
                40.0,  # LH_HFE
                8.0,  # LH_KFE
                50.0,  # RH_HAA
                40.0,  # RH_HFE
                8.0,  # RH_KFE
            ]
        )
        * action_strength / 100.0
    )
    # control_gains = np.ones(12) * 100.0
    control_limits = [(-1.0, 1.0)] * 12

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(
                # os.path.dirname(__file__), "assets/anymal/urdf/anymal_minimal.urdf"
                os.path.dirname(__file__),
                "assets/anymal/urdf/anymal_joint_limits.urdf",
                # os.path.dirname(__file__), "assets/anymal/urdf/anymal_c.urdf"
            ),
            builder,
            floating=True,
            stiffness=85.0,
            damping=2.0,
            armature=0.06,
            contact_ke=2.0e3,
            contact_kd=5.0e2,
            contact_kf=1.0e2,
            contact_mu=0.75,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_visuals=not self.load_visuals,
        )

        self.start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
        self.inv_start_rot = wp.quat_inverse(self.start_rot)

        builder.joint_q[:7] = [
            0.0,
            0.85,
            0.0,
            *self.start_rot,
        ]

        builder.joint_q[7:] = [
            0.03,  # LF_HAA
            0.4,  # LF_HFE
            -0.8,  # LF_KFE
            -0.03,  # RF_HAA
            0.4,  # RF_HFE
            -0.8,  # RF_KFE
            0.03,  # LH_HAA
            -0.4,  # LH_HFE
            0.8,  # LH_KFE
            -0.03,  # RH_HAA
            -0.4,  # RH_HFE
            0.8,  # RH_KFE
        ]

        for i in range(builder.joint_axis_count):
            # builder.joint_act[i] = builder.joint_q[i + 7]
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_FORCE
        #     # builder.joint_q[i + 7] = 0.5 * (builder.joint_limit_lower[i] + builder.joint_limit_upper[i])

        # print(builder.joint_q)
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.device)

        if ZERO_GRAVITY:
            builder.gravity = 0.0

        # print(builder.joint_q)
        # result = builder.sort_joints()
        # result = builder.sort_joints(list(range(builder.joint_count)))
        # print(result)
        # print(builder.joint_q)

        self.torques = wp.zeros(
            self.num_envs * builder.joint_axis_count,
            dtype=wp.float32,
            device=self.device,
        )

        builder.separate_ground_contacts = self.separate_ground_contacts

    def assign_control(
        self,
        actions: wp.array,
        control: wp.sim.Control,
        state: wp.sim.State,
    ):
        super().assign_control(actions, control, state)
        # apply joint stiffness/damping
        self.apply_pd_control(
            control_out=control.joint_act,
            joint_q=state.joint_q,
            joint_qd=state.joint_qd,
            body_q=state.body_q,
        )


if __name__ == "__main__":
    run_env(AnymalEnvironment)
