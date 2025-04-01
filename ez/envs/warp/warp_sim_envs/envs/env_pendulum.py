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

from warp_sim_envs.envs import Environment, run_env, IntegratorType
from warp_sim_envs.utils import angle_normalize


@wp.kernel
def pendulum_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    goal_q: wp.vec2,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    th1 = joint_q[env_id * 2 + 0]
    thdot1 = joint_qd[env_id * 2 + 0]
    th2 = joint_q[env_id * 2 + 1]
    thdot2 = joint_qd[env_id * 2 + 1]
    u1 = joint_act[env_id * 2 + 0]
    u2 = joint_act[env_id * 2 + 1]

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    angle1 = angle_normalize(th1 - goal_q[0])
    angle2 = angle_normalize(th2 - goal_q[1])
    c = (
        angle1**2.0
        + 0.1 * thdot1**2.0
        + angle2**2.0
        + 0.1 * thdot2**2.0
        + (u1 * 1e-4) ** 2.0
        + (u2 * 1e-4) ** 2.0
    )

    wp.atomic_add(cost, env_id, c)

    if terminated:
        terminated[env_id] = abs(thdot1) > 10.0 or abs(thdot2) > 10.0


@wp.kernel(enable_backward=False)
def reset_init_state_pendulum(
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

    if random_reset:
        state = wp.rand_init(seed, env_id)
        joint_q[env_id * dof_q_per_env] = wp.randf(state, -wp.pi, wp.pi)
        joint_q[env_id * dof_q_per_env + 1] = wp.randf(state, -wp.pi, wp.pi)
        joint_qd[env_id * dof_qd_per_env] = wp.randf(state, -2.0, 2.0)
        joint_qd[env_id * dof_qd_per_env + 1] = wp.randf(state, -2.0, 2.0)
    else:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[i]
        


class PendulumEnvironment(Environment):
    robot_name = 'Pendulum'
    sim_name = "env_pendulum"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 16
    sim_substeps_xpbd = 5
    sim_substeps_featherstone = 5

    # sim_substeps_featherstone = 1
    # frame_dt = 1.0 / 300.0
    
    integrator_type = IntegratorType.FEATHERSTONE
    # integrator_type = IntegratorType.XPBD
    # integrator_type = IntegratorType.EULER

    xpbd_settings = dict(iterations=7)

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    activate_ground_plane = False
    use_graph_capture = False

    controllable_dofs = [0, 1]
    control_gains = [5000.0, 5000.0]
    control_limits = [(-1.0, 1.0)] * 2
    use_joint_limits = False

    # whether to use capsules or boxes as link shapes
    use_capsules = True

    goal_q = np.array([np.pi / 2, 0])  # the goal is the upright configuration

    def __init__(self, seed = 42, random_reset = True, **kwargs):
        self.seed = seed
        self.random_reset = random_reset
        super().__init__(**kwargs)

    def create_articulation(self, builder):
        self.joint_type = wp.sim.JOINT_REVOLUTE
        self.chain_length = 2
        self.chain_width = 1.5

        if self.use_joint_limits:
            self.lower = -np.deg2rad(60.0)
            self.upper = np.deg2rad(60.0)
            self.limitd_ke = 1e5
            self.limitd_kd = 1e2
        else:
            self.lower = -2 * np.pi
            self.upper = 2 * np.pi
            self.limitd_ke = 0.0
            self.limitd_kd = 0.0

        self.target = np.deg2rad(15)
        self.target_ke = 0.0  # 1e4
        self.target_kd = 0.0

        # shape_ke = 2.0e4
        # shape_kd = 1.0e3
        # shape_kf = 1.0e4
        shape_ke = 2.0e4
        shape_kd = 1.0e3
        shape_kf = 1.0e4

        builder.set_ground_plane(
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
        )

        for i in range(self.chain_length):
            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 2.0, 1.0], wp.quat_identity())
            else:
                parent = builder.joint_count - 1
                parent_joint_xform = wp.transform(
                    [self.chain_width, 0.0, 0.0], wp.quat_identity()
                )

            # NOTE: the origin of the body will be ignored in an articulation
            # Reference of the articulaion in warp: 
            # https://nvidia.github.io/warp/modules/sim.html#forward-inverse-kinematics
                
            # create body
            b = builder.add_body(
                origin=wp.transform([i, 0.0, 1.0], wp.quat_identity()), armature=0.1
            )

            # create shape
            if self.use_capsules:
                builder.add_shape_capsule(
                    pos=(self.chain_width * 0.5, 0.0, 0.0),
                    half_height=self.chain_width * 0.5,
                    radius=0.1,
                    up_axis=0,
                    density=500.0,
                    body=b,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                )
            else:
                builder.add_shape_box(
                    pos=(self.chain_width * 0.5, 0.0, 0.0),
                    hx=self.chain_width * 0.5,
                    hy=0.1,
                    hz=0.1,
                    density=500.0,
                    body=b,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                )

            if self.joint_type == wp.sim.JOINT_REVOLUTE:
                if i == 0:
                    builder.add_joint_revolute(
                        parent=parent,
                        child=b,
                        axis=(0.0, 0.0, 1.0),
                        parent_xform=parent_joint_xform,
                        limit_lower=self.lower,
                        limit_upper=self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                    )
                else:
                    builder.add_joint_revolute(
                        parent=parent,
                        child=b,
                        axis=(0.0, 0.0, 1.0),
                        parent_xform=parent_joint_xform,
                        limit_lower=self.lower,
                        limit_upper=self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                    )

            elif self.joint_type == wp.sim.JOINT_UNIVERSAL:
                builder.add_joint_universal(
                    parent=parent,
                    child=b,
                    axis_0=wp.sim.JointAxis(
                        (1.0, 0.0, 0.0),
                        self.lower,
                        self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                    ),
                    axis_1=wp.sim.JointAxis(
                        (0.0, 0.0, 1.0),
                        self.lower,
                        self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                        target=self.target,
                        target_ke=self.target_ke,
                        target_kd=self.target_kd,
                    ),
                    parent_xform=parent_joint_xform,
                )

            elif self.joint_type == wp.sim.JOINT_BALL:
                builder.add_joint_ball(
                    parent=parent,
                    child=b,
                    parent_xform=parent_joint_xform,
                    armature=10.0,
                )

            elif self.joint_type == wp.sim.JOINT_FIXED:
                builder.add_joint_fixed(
                    parent=parent,
                    child=b,
                    parent_xform=parent_joint_xform,
                )

            elif self.joint_type == wp.sim.JOINT_COMPOUND:
                builder.add_joint_compound(
                    parent=parent,
                    child=b,
                    axis_0=wp.sim.JointAxis(
                        (1.0, 0.0, 0.0),
                        self.lower,
                        self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                    ),
                    axis_1=wp.sim.JointAxis(
                        (0.0, 1.0, 0.0),
                        self.lower,
                        self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                    ),
                    axis_2=wp.sim.JointAxis(
                        (0.0, 0.0, 1.0),
                        self.lower,
                        self.upper,
                        limit_ke=self.limitd_ke,
                        limit_kd=self.limitd_kd,
                        target=self.target,
                        target_ke=self.target_ke,
                        target_kd=self.target_kd,
                    ),
                    parent_xform=parent_joint_xform,
                )

            elif self.joint_type == wp.sim.JOINT_D6:
                builder.add_joint_d6(
                    parent=parent,
                    child=b,
                    angular_axes=[
                        wp.sim.JointAxis(
                            (1.0, 0.0, 0.0),
                            self.lower,
                            self.upper,
                            limit_ke=self.limitd_ke,
                            limit_kd=self.limitd_kd,
                        ),
                        wp.sim.JointAxis(
                            (0.0, 1.0, 0.0),
                            self.lower,
                            self.upper,
                            limit_ke=self.limitd_ke,
                            limit_kd=self.limitd_kd,
                        ),
                        wp.sim.JointAxis(
                            (0.0, 0.0, 1.0),
                            self.lower,
                            self.upper,
                            limit_ke=self.limitd_ke,
                            limit_kd=self.limitd_kd,
                            target=self.target,
                            target_ke=self.target_ke,
                            target_kd=self.target_kd,
                        ),
                    ],
                    parent_xform=parent_joint_xform,
                )

        # builder.joint_q[:] = [-np.pi / 2, 0.0]
        builder.joint_q[:] = [0., 0.0]

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
            pendulum_cost,
            dim=self.num_envs,
            inputs=[
                state.joint_q, 
                state.joint_qd, 
                control.joint_act, 
                self.goal_q
            ],
            outputs=[cost, terminated],
            device=self.device,
        )

    def reset_envs(self, env_ids: wp.array = None):
        wp.launch(
            reset_init_state_pendulum,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed,
                self.random_reset,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_q,
                self.model.joint_qd
            ],
            outputs=[self.state.joint_q, self.state.joint_qd],
            device=self.device,
        )
        self.seed += self.num_envs

        # update maximal coordinates after reset
        if env_ids is None or env_ids.numpy().any():
            if self.uses_generalized_coordinates:
                wp.sim.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, None, self.state)
                

if __name__ == "__main__":
    run_env(PendulumEnvironment)
