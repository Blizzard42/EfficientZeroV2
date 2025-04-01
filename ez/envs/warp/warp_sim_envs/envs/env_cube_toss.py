# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Cube tossing environment
###########################################################################

import os
import torch
import warp as wp
import warp.sim

from warp_sim_envs.envs import Environment, IntegratorType, RenderMode


# see https://github.com/DAIRLab/contact-nets/blob/main/README.md
METER_SCALE = 0.0524

# see https://github.com/DAIRLab/contact-nets/blob/main/data/params_processed/experiment.json
SAMPLING_DT = 0.006756756756756757

TRAJECTORY_PATH = (
    "C:/Users/eric-/source/repos/contact-nets/data/tosses_processed/{ti}.pt"
)

NUM_REAL_TRAJECTORIES = 570


@wp.func
def has_ground_penetration(pos: wp.vec3, quat: wp.quat):
    margin = 1e-3
    tf = wp.transform(pos, quat)
    # fmt: off
    if wp.transform_point(tf, wp.vec3(METER_SCALE, METER_SCALE, METER_SCALE))[2] < margin:
        return True
    if wp.transform_point(tf, wp.vec3(-METER_SCALE, METER_SCALE, METER_SCALE))[2] < margin:
        return True
    if wp.transform_point(tf, wp.vec3(METER_SCALE, -METER_SCALE, METER_SCALE))[2] < margin:
        return True
    if (wp.transform_point(tf, wp.vec3(-METER_SCALE, -METER_SCALE, METER_SCALE))[2] < margin):
        return True
    if wp.transform_point(tf, wp.vec3(METER_SCALE, METER_SCALE, -METER_SCALE))[2] < margin:
        return True
    if (wp.transform_point(tf, wp.vec3(-METER_SCALE, METER_SCALE, -METER_SCALE))[2] < margin):
        return True
    if (wp.transform_point(tf, wp.vec3(METER_SCALE, -METER_SCALE, -METER_SCALE))[2] < margin):
        return True
    if (wp.transform_point(tf, wp.vec3(-METER_SCALE, -METER_SCALE, -METER_SCALE))[2] < margin):
        return True
    return False
    # fmt: on


@wp.kernel(enable_backward=False)
def reset_cube_old(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    # initial_qs: wp.array(dtype=wp.float32, ndim=2),
    # initial_qds: wp.array(dtype=wp.float32, ndim=2),
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
            env_id * dof_qd_per_env + i
        ]

    if random_reset:
        random_state = wp.rand_init(seed, env_id)

        resample = bool(True)
        while resample:
            # randomize base position
            pos = wp.vec3(
                wp.randf(random_state, -20.0 * METER_SCALE, 20.0 * METER_SCALE),
                wp.randf(random_state, -20.0 * METER_SCALE, 20.0 * METER_SCALE),
                wp.randf(random_state, 0.9 * METER_SCALE, 5.0 * METER_SCALE),
            )
            joint_q[env_id * dof_q_per_env + 0] = pos[0]
            joint_q[env_id * dof_q_per_env + 1] = pos[1]
            joint_q[env_id * dof_q_per_env + 2] = pos[2]

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

            # check if the cube is penetrating the ground
            resample = has_ground_penetration(pos, quat)

        # randomize base angular velocities
        joint_qd[env_id * dof_qd_per_env + 0] = wp.randf(random_state, -2.0, 2.0)
        joint_qd[env_id * dof_qd_per_env + 1] = wp.randf(random_state, -2.0, 2.0)
        joint_qd[env_id * dof_qd_per_env + 2] = wp.randf(random_state, -2.0, 2.0)
        # for i in range(3):
        #     joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[
        #         env_id * dof_qd_per_env + i
        #     ] + 0.25 * wp.randf(random_state, -1.0, 1.0)

        # randomize base linear velocities
        # for i in range(3, 6):
        #     joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[
        #         env_id * dof_qd_per_env + i
        #     ] + 0.1 * wp.randf(random_state, -1.0, 1.0)
        joint_qd[env_id * dof_qd_per_env + 3] = wp.randf(random_state, -3.5, 3.5)
        joint_qd[env_id * dof_qd_per_env + 4] = wp.randf(random_state, -3.5, 3.5)
        joint_qd[env_id * dof_qd_per_env + 5] = wp.randf(random_state, -1.0, 0.2)

@wp.kernel(enable_backward=False)
def reset_cube(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    # initial_qs: wp.array(dtype=wp.float32, ndim=2),
    # initial_qds: wp.array(dtype=wp.float32, ndim=2),
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
            env_id * dof_qd_per_env + i
        ]

    if random_reset:
        random_state = wp.rand_init(seed, env_id)

        resample = bool(True)
        while resample:
            # randomize base position
            pos = wp.vec3(
                wp.randf(random_state, -20.0 * METER_SCALE, 20.0 * METER_SCALE),
                wp.randf(random_state, -20.0 * METER_SCALE, 20.0 * METER_SCALE),
                wp.randf(random_state, 0.9 * METER_SCALE, 5.0 * METER_SCALE),
            )
            joint_q[env_id * dof_q_per_env + 0] = pos[0]
            joint_q[env_id * dof_q_per_env + 1] = pos[1]
            joint_q[env_id * dof_q_per_env + 2] = pos[2]

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

            # check if the cube is penetrating the ground
            resample = has_ground_penetration(pos, quat)

        # randomize base angular and linear velocities
        pos_base = wp.vec3(joint_q[dof_q_per_env * env_id + 0],
                           joint_q[dof_q_per_env * env_id + 1],
                           joint_q[dof_q_per_env * env_id + 2])
        
        ang_vel_base_body = wp.vec3(
            wp.randf(random_state, -10., 12.),
            wp.randf(random_state, -10., 8.),
            wp.randf(random_state, -7., 4.)
        )
        lin_vel_base_body = wp.vec3(
            3.5 * wp.randf(random_state, -1., 1.),
            3.5 * wp.randf(random_state, -1., 1.),
            wp.randf(random_state, -1., 0.2)
        )

        ang_vel_base_world = wp.quat_rotate(quat, ang_vel_base_body)
        lin_vel_base_world = wp.cross(pos_base, wp.quat_rotate(quat, ang_vel_base_body)) + wp.quat_rotate(quat, lin_vel_base_body)
        for i in range(3):
            joint_qd[env_id * dof_qd_per_env + i] = ang_vel_base_world[i]
            # joint_qd[env_id * dof_qd_per_env + i] = ang_vel_base_body[i]
        for i in range(3):
            joint_qd[env_id * dof_qd_per_env + 3 + i] = lin_vel_base_world[i]
            # joint_qd[env_id * dof_qd_per_env + 3 + i] = lin_vel_base_body[i]

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

class CubeTossingEnvironment(Environment):
    robot_name = "CubeToss"
    sim_name = "env_cube_toss"

    episode_duration = 0.5

    opengl_render_settings = dict(scaling=5.0, far_plane=1000.0)
    usd_render_settings = dict(scaling=10.0)

    sim_substeps_euler = 20
    sim_substeps_featherstone = 20
    sim_substeps_xpbd = 5

    integrator_type = IntegratorType.FEATHERSTONE
    # integrator_type = IntegratorType.EULER
    # integrator_type = IntegratorType.XPBD

    featherstone_settings = {"angular_damping": 0.0}

    frame_dt = SAMPLING_DT

    enable_friction = True

    # render_mode = RenderMode.USD

    featherstone_update_mass_matrix_once_per_step = False
    handle_collisions_once_per_step = False

    activate_ground_plane = True

    use_graph_capture = False

    num_envs = 1

    up_axis = "Z"

    show_rigid_contact_points = True
    # show_rigid_contact_points = False
    contact_points_radius = 0.001

    def __init__(self, seed=42, random_reset=True, **kwargs):
        self.seed = seed
        self.random_reset = random_reset
        super().__init__(**kwargs)

        self.initial_qs = None
        self.initial_qds = None

        # if os.path.exists(TRAJECTORY_PATH.format(ti=0)):
        #     # load initial states from the dataset
        #     qs = []
        #     qds = []
        #     for i in range(570):
        #         trajectory = torch.load(TRAJECTORY_PATH.format(ti=i))[0]
        #         q, qd = env.torch_state_to_q_qd(trajectory[0])
        #         qs.append(q.cpu().numpy())
        #         qds.append(qd.cpu().numpy())

        #     self.initial_qs = wp.array(qs, dtype=wp.float32)
        #     self.initial_qds = wp.array(qds, dtype=wp.float32)

        #     print(f"Successfully loaded {len(qs)} initial states.")
        # else:
        #     print("Could not find trajectory data. Consider updating TRAJECTORY_PATH from https://github.com/DAIRLab/contact-nets/")

    def create_articulation(self, builder):
        shape_ke = 1e4
        shape_kd = 1e3
        # disable friction
        if self.enable_friction:
            shape_kf = 50.0
        else:
            shape_kf = 0.0
        shape_mu = 0.15

        b = builder.add_body(name="cube")
        if True:
            builder.add_shape_box(
                b,
                hx=METER_SCALE,
                hy=METER_SCALE,
                hz=METER_SCALE,
                density=1000.0,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu,
            )
        else:
            builder.add_shape_sphere(
                b,
                radius=METER_SCALE,
                density=1000.0,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu,
            )
        # builder.body_mass[b] = 0.37
        # inertia = 0.2950002913583124 * METER_SCALE**2
        # # fmt: off
        # builder.body_inertia[b] = wp.mat33(
        #     inertia, 0.0, 0.0,
        #     0.0, inertia, 0.0,
        #     0.0, 0.0, inertia,
        # )
        # fmt: on
        builder.add_joint_free(child=b, parent=-1)
        builder.joint_q[2] = METER_SCALE * 1.5

        builder.set_ground_plane(
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
            mu=shape_mu,
        )

    @staticmethod
    def torch_state_to_q_qd(torch_state):
        import torch
        # format of ContactNets dataset:
        # position (3), quaternion (4), velocity (3), angular velocity (3), control (6)
        qi = [0, 1, 2, 4, 5, 6, 3]
        qdi = [10, 11, 12, 7, 8, 9]
        if torch_state.dim() == 1:
            q = torch_state[qi].float()
            qd = torch_state[qdi].float()
            if False:
                # check frame of reference of velocities
                q_np = q.cpu().numpy()
                qd_np = qd.cpu().numpy()
                pos = wp.vec3(*q_np[0:3])
                rot = wp.quat(*q_np[3:7])
                tf = wp.transform(pos, rot)
                penetrating = has_ground_penetration(pos, rot)
                # if penetrating:
                #     print("penetrating", penetrating)
                ang_vel = wp.vec3(*qd_np[0:3])
                lin_vel = wp.vec3(*qd_np[3:6])

                if False:
                    # convert world to body velocities
                    rot_inv = wp.quat_inverse(rot)
                    
                    ang_vel_body = wp.quat_rotate(rot_inv, ang_vel)
                    lin_vel_body = wp.quat_rotate(rot_inv, lin_vel - wp.cross(ang_vel, pos))
                    qd_np[0:3] = ang_vel_body
                    qd_np[3:6] = lin_vel_body
                if False:
                    # convert body to world velocities
                    ang_vel_world = wp.quat_rotate(rot, ang_vel)
                    lin_vel_world = wp.quat_rotate(rot, lin_vel) + wp.cross(pos, ang_vel_world)
                    # qd_np[0:3] = ang_vel_world * METER_SCALE
                    # qd_np[0:3] = ang_vel * METER_SCALE
                    # qd_np[3:6] = lin_vel_world # * METER_SCALE
                import torch
                qd = torch.tensor(qd_np, dtype=torch.float32)
            if True:
                x, quat = q[0:3], q[3:7]
                ang_vel_body, lin_vel = qd[0:3], qd[3:6]
                ang_vel_world = quat_rotate(quat.unsqueeze(0), ang_vel_body.unsqueeze(0)).squeeze(0)
                lin_vel_world = lin_vel + torch.cross(x, ang_vel_world)
                qd[0:3] = ang_vel_world
                qd[3:6] = lin_vel_world
        else:
            q = torch_state[:, qi].float()
            qd = torch_state[:, qdi].float()

            if True:
                x, quat = q[:, 0:3], q[:, 3:7]
                ang_vel_body, lin_vel = qd[:, 0:3], qd[:, 3:6]
                ang_vel_world = quat_rotate(quat, ang_vel_body)
                lin_vel_world = lin_vel + torch.cross(x, ang_vel_world, dim = -1)
                qd[:, 0:3] = ang_vel_world
                qd[:, 3:6] = lin_vel_world
        return q, qd

    def set_torch_state(
        self,
        torch_state,
        state: wp.sim.State = None,
        eval_fk: bool = True,
    ):
        # format of the state in ContactNets is as follows:
        # position (3), quaternion (4), velocity (3), angular velocity (3), control (6)
        if state is None:
            state = self.state
        q, qd = self.torch_state_to_q_qd(torch_state)
        state.joint_q.assign(wp.from_torch(q))
        state.joint_qd.assign(wp.from_torch(qd))
        if eval_fk:
            wp.sim.eval_fk(self.model, state.joint_q, state.joint_qd, None, state)

    @staticmethod
    def load_trajectory(
        index: int,
        apply_scaling: bool = True,
    ):
        import torch

        trajectory = torch.load(TRAJECTORY_PATH.format(ti=index), weights_only=True)[0]
        if apply_scaling:
            trajectory[:, :3] *= METER_SCALE
            trajectory[:, 7:10] *= METER_SCALE
            # also need to scale angula velocity
            # trajectory[:, 10:13] *= METER_SCALE
        return trajectory

    def reset_envs(self, env_ids: wp.array = None):
        """Reset environments where env_ids buffer indicates True. Resets all envs if env_ids is None."""
        wp.launch(
            reset_cube,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed,
                self.random_reset,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_q,
                self.model.joint_qd,
                # self.initial_qs,
                # self.initial_qds,
            ],
            outputs=[
                self.state.joint_q,
                self.state.joint_qd,
            ],
            device=self.device,
        )
        self.seed += self.num_envs

        # print("reset envs", self.state.joint_q.numpy())

        # update maximal coordinates after reset
        if env_ids is None or env_ids.numpy().any():
            if self.uses_generalized_coordinates:
                wp.sim.eval_fk(
                    self.model,
                    self.state.joint_q,
                    self.state.joint_qd,
                    None,
                    self.state,
                )

    def custom_render(self, render_state, renderer):
        if self.render_mode == RenderMode.OPENGL:
            cam_pos = wp.vec3(0., 2., 15.)
            with wp.ScopedTimer("update_view_matrix", color=0x663300, active=self.enable_timers):
                self.renderer.update_view_matrix(cam_pos=cam_pos)


if __name__ == "__main__":
    env = CubeTossingEnvironment(num_envs=10)
    # env.run()
    # env.reset()

    if env.use_graph_capture:
        with wp.ScopedCapture() as capture:
            env.update()
        graph = capture.graph

    running = True

    while running:
        env.reset()
        for f in range(env.episode_frames):
            if env.use_graph_capture:
                with wp.ScopedTimer(
                    "simulation_graph", color="green", active=env.enable_timers
                ):
                    wp.capture_launch(graph)
                env.sim_time += env.frame_dt
                env.sim_step += env.sim_substeps
            elif not env.requires_grad or env.sim_step < env.sim_steps:
                with wp.ScopedTimer("update", color="green", active=env.enable_timers):
                    env.update()

            env.render()
            if env.render_mode == RenderMode.OPENGL and env.renderer.has_exit:
                running = False
                break

    # state = env.state

    # import torch

    # for ti in range(5):
    #     print(f"Playing back trajectory {ti}")

    #     trajectory = torch.load(TRAJECTORY_PATH.format(ti=ti))[0]
    #     # convert linear components to meters
    #     trajectory[:, :3] *= METER_SCALE
    #     trajectory[:, 7:10] *= METER_SCALE

    #     for i in range(trajectory.shape[0]):
    #         env.sim_time += SAMPLING_DT
    #         env.set_torch_state(trajectory[i], state)
    #         env.render(state)

    # env.renderer.save()
