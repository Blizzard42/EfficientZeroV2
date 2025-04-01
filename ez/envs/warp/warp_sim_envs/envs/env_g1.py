# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Pendulum environment
#
# Shows how to set up a simulation of a rigid-body Hopper articulation based on
# the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import numpy as np

import warp as wp
import warp.sim

from warp_sim_envs.envs import Environment, run_env, IntegratorType
from warp_sim_envs.envs.env_humanoid import calculate_cost, calculate_observations


class HumanoidG1Environment(Environment):
    sim_name = "env_g1"
    env_offset = (2.0, 0.0, 2.0)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 16
    sim_substeps_xpbd = 5
    sim_substeps_featherstone = 20

    # integrator_type = IntegratorType.FEATHERSTONE
    integrator_type = IntegratorType.XPBD
    # integrator_type = IntegratorType.EULER

    xpbd_settings = dict(iterations=7)

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    activate_ground_plane = True
    # use_graph_capture = False

    num_envs = 4

    control_gains = (
        np.array(
            [
                200,  # left_hip_pitch_joint
                200,  # left_hip_roll_joint
                600,  # left_hip_yaw_joint
                400,  # left_knee_joint
                100,  # left_ankle_pitch_joint
                100,  # left_ankle_roll_joint
                200,  # right_hip_pitch_joint
                200,  # right_hip_roll_joint
                600,  # right_hip_yaw_joint
                400,  # right_knee_joint
                200,  # right_ankle_pitch_joint
                200,  # right_ankle_roll_joint
                300,  # waist_yaw_joint
                100,  # left_shoulder_pitch_joint
                100,  # left_shoulder_roll_joint
                100,  # left_shoulder_yaw_joint
                200,  # left_elbow_joint
                100,  # left_wrist_roll_joint
                100,  # right_shoulder_pitch_joint
                100,  # right_shoulder_roll_joint
                100,  # right_shoulder_yaw_joint
                200,  # right_elbow_joint
                100,  # right_wrist_roll_joint
            ]
        )
        * 0.2
    )
    control_dim = len(control_gains)
    control_limits = [(-1.0, 1.0)] * control_dim
    controllable_dofs = np.arange(control_dim)

    def __init__(self, seed=42, **kwargs):
        self.seed = seed
        super().__init__(**kwargs)

    def create_articulation(self, builder):
        if True:
            wp.sim.parse_mjcf(
                r"C:\Users\eric-\Projects\unitree_ros\robots\g1_description\g1_23dof.xml",
                builder,
                xform=wp.transform(wp.vec3(0.0, -0.5, -0.8), wp.quat_from_axis_angle((1.0, 0.0, 0.0), wp.pi/2.0)),
                stiffness=0.0,
                damping=0.001,
                armature=0.001,
                contact_ke=1.0e1,
                contact_kd=1.0e0,
                contact_kf=1.0e0,
                contact_mu=1.0,
                limit_ke=1.0e3,
                limit_kd=1.0e1,
                enable_self_collisions=False,
            )
        else:
            wp.sim.parse_urdf(
                # r"C:\Users\eric-\Projects\unitree_ros\robots\g1_description\g1_23dof.urdf",
                r"C:\Users\eric-\Projects\unitree_ros\robots\g1_description\g1_lower_body.urdf",
                builder,
                floating=False,
                armature=0.001,
                enable_self_collisions=False,
                parse_visuals_as_colliders=True,
                xform=wp.transform(wp.vec3(0.0, 0.8, 0.0), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -wp.pi/2.0)),
                contact_ke=1.0e0,
                contact_kd=0.0e0,
                contact_kf=1.0e0,
                contact_mu=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
                # ensure we use force control
                stiffness=0.0,
                damping=0.0,
            )
        builder.collapse_fixed_joints()

        builder.joint_q[0] = -0.5
        builder.joint_act[0] = -0.5
        
        # for i in range(builder.joint_count):
        #     q = builder.joint_limit_lower[i] + 0.5 * (builder.joint_limit_upper[i] - builder.joint_limit_lower[i])
        #     builder.joint_q[i] = q
        #     builder.joint_act[i] = q

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


if __name__ == "__main__":
    run_env(HumanoidG1Environment)
