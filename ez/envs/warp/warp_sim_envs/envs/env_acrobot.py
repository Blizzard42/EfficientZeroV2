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
from warp_sim_envs.utils import angle_normalize


# @wp.kernel
# def acrobot_cost(
#     joint_q: wp.array(dtype=wp.float32),
#     joint_qd: wp.array(dtype=wp.float32),
#     joint_act: wp.array(dtype=wp.float32),
#     goal_q: wp.array(dtype=wp.vec2),
#     # outputs
#     cost: wp.array(dtype=wp.float32),
#     terminated: wp.array(dtype=wp.bool),
# ):
#     env_id = wp.tid()

#     th1 = joint_q[env_id * 2 + 0]
#     thdot1 = joint_qd[env_id * 2 + 0]
#     th2 = joint_q[env_id * 2 + 1]
#     thdot2 = joint_qd[env_id * 2 + 1]
#     u = joint_act[env_id * 2 + 1]
#     goal = goal_q[env_id]

#     # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
#     angle1 = angle_normalize(th1 - goal[0])
#     angle2 = angle_normalize(th2 - goal[1])
#     c = (
#         angle1**2.0
#         + angle2**2.0
#     )

#     wp.atomic_add(cost, env_id, c)

#     if terminated:
#         terminated[env_id] = abs(thdot1) > 10.0 or abs(thdot2) > 10.0

@wp.kernel
def acrobot_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    th1 = joint_q[env_id * 2 + 0]
    th2 = joint_q[env_id * 2 + 1]
    thdot1 = joint_qd[env_id * 2 + 0]
    thdot2 = joint_qd[env_id * 2 + 1]

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    angle1 = th1 + wp.pi / 2.
    angle2 = th2

    c = 1.
    if abs(thdot1) > 10 or abs(thdot2) > 10:
        failed = True
        c += 300.
    else:
        failed = False

    wp.atomic_add(cost, env_id, c)

    terminated[env_id] = -wp.cos(angle1) - wp.cos(angle1 + angle2) > 1.8 or failed

@wp.kernel(enable_backward=False)
def reset_init_state_acrobot(
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
        random_state = wp.rand_init(seed, env_id)
        joint_q[env_id * dof_q_per_env] = wp.randf(random_state, -wp.pi, wp.pi)
        joint_q[env_id * dof_q_per_env + 1] = wp.randf(random_state, -wp.pi, wp.pi)
        joint_qd[env_id * dof_qd_per_env] = wp.randf(random_state, -1.0, 1.0)
        joint_qd[env_id * dof_qd_per_env + 1] = wp.randf(random_state, -1.0, 1.0)
    else:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[i]

@wp.kernel
def compute_observations_acrobot(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    obs: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    obs[tid, 0] = wp.sin(joint_q[tid * dof_q_per_env + 0])
    obs[tid, 1] = wp.cos(joint_q[tid * dof_q_per_env + 0])
    obs[tid, 2] = wp.sin(joint_q[tid * dof_q_per_env + 1])
    obs[tid, 3] = wp.cos(joint_q[tid * dof_q_per_env + 1])
    for i in range(dof_qd_per_env):
        obs[tid, i + 4] = joint_qd[tid * dof_qd_per_env + i]


class AcrobotEnvironment(Environment):
    robot_name = 'Pendulum'
    sim_name = "env_acrobot"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_featherstone = 5

    integrator_type = IntegratorType.FEATHERSTONE

    activate_ground_plane = False

    controllable_dofs = [1]
    control_gains = [1500.]
    control_limits = [(-1.0, 1.0)]

    def __init__(self, seed = 42, random_reset = True, **kwargs):
        self.seed = seed
        self.random_reset = random_reset
        super().__init__(**kwargs)

    def create_articulation(self, builder):
        self.joint_type = wp.sim.JOINT_REVOLUTE
        self.chain_length = 2
        self.chain_width = 1.5

        self.lower = -2 * np.pi
        self.upper = 2 * np.pi
        self.limitd_ke = 0.0
        self.limitd_kd = 0.0
        
        shape_ke = 1.0e3
        shape_kd = 1.0e2
        shape_kf = 1.0e3

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

            # create body
            b = builder.add_body(
                origin=wp.transform([i, 0.0, 1.0], wp.quat_identity()), armature=0.1
            )

            # create shape
            builder.add_shape_box(
                pos=(self.chain_width * 0.5, 0.0, 0.0),
                hx=self.chain_width * 0.5,
                hy=0.1,
                hz=0.1,
                density=500.0,
                body=b,
            )

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

        builder.joint_q[:] = [-np.pi / 2, 0.0]

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
            acrobot_cost,
            dim=self.num_envs,
            inputs=[
                state.joint_q, 
                state.joint_qd, 
                control.joint_act
            ],
            outputs=[cost, terminated],
            device=self.device,
        )

    def reset_envs(self, env_ids: wp.array = None):
        wp.launch(
            reset_init_state_acrobot,
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
                

    @property
    def observation_dim(self):
        return self.dof_q_per_env * 2 + self.dof_qd_per_env
    
    def compute_observations(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        wp.launch(
            compute_observations_acrobot,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[observations],
            device=self.device,
        )

if __name__ == "__main__":
    run_env(AcrobotEnvironment)
