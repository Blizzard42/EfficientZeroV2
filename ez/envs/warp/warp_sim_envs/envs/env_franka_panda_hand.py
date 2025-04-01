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

wp.config.verify_cuda = True

from warp_sim_envs.envs import Environment, run_env, IntegratorType, RenderMode
from warp_sim_envs.utils import generate_pd_control, compute_body_jacobian


def quaternion_from_vectors(a, b):
    """
    Compute the quaternion that rotates vector a onto vector b.
    Both a and b are 3D NumPy arrays (shape = (3,)).

    Returns:
        A NumPy array [w, x, y, z], representing the quaternion.
    """
    # Normalize the input vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)

    # Dot product and cross product
    dot = np.dot(a_norm, b_norm)

    # If vectors are almost identical, return identity quaternion
    if np.isclose(dot, 1.0):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # If vectors are nearly opposite, find an orthogonal vector for the rotation axis
    if np.isclose(dot, -1.0):
        # Try a simple orthogonal candidate
        ortho = np.cross(a_norm, np.array([1.0, 0.0, 0.0]))
        # If this is too close to zero, pick another axis
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(a_norm, np.array([0.0, 1.0, 0.0]))
        ortho = ortho / np.linalg.norm(ortho)
        # 180-degree rotation around 'ortho' axis => quaternion = [0, ortho.x, ortho.y, ortho.z]
        return np.array([0.0, ortho[0], ortho[1], ortho[2]], dtype=np.float64)

    # General case: angle = arccos(a · b), axis = a x b
    axis = np.cross(a_norm, b_norm)
    axis_len = np.linalg.norm(axis)
    axis = axis / axis_len
    angle = np.arccos(dot)

    half_angle = angle / 2.0
    w = np.cos(half_angle)
    sin_half = np.sin(half_angle)
    x, y, z = axis * sin_half

    return np.array([w, x, y, z], dtype=np.float64)


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


@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    bodies_per_env: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    env_id = wp.tid()
    tf = body_q[bodies_per_env * env_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[env_id] = wp.spatial_vector(
        ang_diff[0], ang_diff[1], ang_diff[2], pos_diff[0], pos_diff[1], pos_diff[2]
    )


@wp.func
def allclose(a: wp.vec3, b: wp.vec3):
    # TODO support wp.func default args in Python context
    rtol = 1e-5
    atol = 1e-8
    return (
        wp.abs(a[0] - b[0]) <= (atol + rtol * wp.abs(b[0]))
        and wp.abs(a[1] - b[1]) <= (atol + rtol * wp.abs(b[1]))
        and wp.abs(a[2] - b[2]) <= (atol + rtol * wp.abs(b[2]))
    )


@wp.func
def vec_rotation(x: float, y: float, z: float) -> wp.transform:
    """
    Converts plane coordinates given by the plane normal and its offset along the normal to a transform.
    """
    normal = wp.normalize(wp.vec3(x, y, z))
    if allclose(normal, wp.vec3(0.0, 0.0, 1.0)):
        # no rotation necessary
        return wp.quat(0.0, 0.0, 0.0, 1.0)
    elif allclose(normal, wp.vec3(0.0, 0.0, -1.0)):
        # 180 degree rotation around x-axis
        return wp.quat(1.0, 0.0, 0.0, 0.0)
    else:
        c = wp.cross(wp.vec3(0.0, 0.0, 1.0), normal)
        angle = wp.asin(wp.length(c))
        # adjust for arcsin ambiguity
        if wp.dot(normal, wp.vec3(0.0, 0.0, 1.0)) < 0:
            angle = wp.pi - angle
        axis = c / wp.length(c)
        return wp.quat_from_axis_angle(axis, angle)


class FrankaPandaEnvironment(Environment):
    robot_name = "Franka"
    sim_name = "env_franka_panda"
    env_offset = (2.0, 0.0, 2.0)
    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)
    usd_save_as_usda = True

    sim_substeps_euler = 32
    sim_substeps_featherstone = 1
    sim_substeps_xpbd = 10

    episode_duration = 20.0

    use_graph_capture = False

    num_rigid_contacts_per_env = 0

    activate_ground_plane = False

    integrator_type = IntegratorType.FEATHERSTONE
    render_mode = RenderMode.USD

    show_joints = False

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

    has_gripper = True

    # plot_joint_coords = True
    plot_joint_coords_qd = False
    use_graph_capture = False

    num_envs = 1

    # joint_damping = 0.5
    joint_damping = 0.0

    gravity = 0.0  # disable gravity for franka

    target_pos_bounds = wp.array(
        [[0.3, 0.7], [0.034, 0.184], [-0.35, 0.35]], dtype=wp.float32
    )  # bounds within which to select target position for subsequent movement of gripper using RL policy ([[x_min, x_max], [y_min, y_max], [z_min, z_max]])

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
            limit_ke=0.0,
            limit_kd=0.0,
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
                has_ground_collision=False,
            )

        self.targets = np.array(
            [
                # gripper transform (3D position, 4D quaternion), gripper open (1) or closed (0)
                [0.0, 0.25, 0.6, *vec_rotation(0.0, -1.0, 0.0), 0.0],
                [-0.6, 0.35, 0.0, *vec_rotation(-1.0, 0.0, 0.0), 1.0],
                [
                    0.0,
                    0.15,
                    -0.6,
                    *wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi),
                    0.0,
                ],
                [0.6, 0.35, 0.0, *vec_rotation(1.0, -1.0, -2.0), 1.0],
            ],
            dtype=np.float32,
        )
        self.target = self.targets[0]

        for i in range(builder.joint_count):
            builder.joint_act[i] = 0.0
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_FORCE

    def before_step(
        self,
        state: wp.sim.State,
        next_state: wp.sim.State,
        control: wp.sim.Control,
        eval_collisions: bool = True,
    ):
        # wp.launch(
        #     apply_damping,
        #     dim=self.num_envs * self.dof_qd_per_env,
        #     inputs=[self.state.joint_qd, self.joint_damping, self.model.body_mass],
        #     outputs=[self.control.joint_act],
        #     device=self.device,
        # )

        assert self.num_envs == 1, "Only one environment supported for now"

        ee_delta = wp.empty(self.num_envs, dtype=wp.spatial_vector, device=self.device)

        wp.launch(
            compute_ee_delta,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_env,
                wp.transform(*self.target[:7]),
            ],
            outputs=[ee_delta],
            device=self.device,
        )

        def ease(t, start, end, duration):
            if t > duration:
                return end
            c = end - start
            return -c * (t / duration) * (t / duration - 2.0) + start

        include_rotation = True
        switch_duration = 3.0
        self.target = self.targets[
            int(self.sim_time / switch_duration) % len(self.targets)
        ]

        if include_rotation:
            delta_target = ee_delta.numpy()[0]
            # return
        else:
            delta_target = ee_delta.numpy()[0, :3]

        J = compute_body_jacobian(
            self.model,
            state.joint_q,
            state.joint_qd,
            body_id=self.endeffector_id,
            offset=self.endeffector_offset,
            include_rotation=include_rotation,
        )

        J_inv = np.linalg.pinv(J)

        # 2. Compute null-space projector
        #    I is size [num_joints x num_joints]
        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state.joint_q.numpy()

        # 3. Define a desired "elbow-up" reference posture
        #    (For example, one that keeps joint 2 or 3 above a certain angle.)
        #    Adjust indices and angles to your robot's kinematics.
        q_des = q.copy()
        q_des[1:] = self.model.joint_q.numpy()[
            1:
        ]  # e.g., set elbow joint around 1 rad to keep it up

        # 4. Define a null-space velocity term pulling joints toward q_des
        #    K_null is a small gain so it doesn't override main task
        K_null = 1.0
        delta_q_null = K_null * (q_des - q)

        # 5. Combine primary task and null-space controller
        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Apply gripper finger control
        delta_q[-2] = self.target[-1] * 0.04 - q[-2]
        delta_q[-1] = self.target[-1] * 0.04 - q[-1]

        # delta_q = J_inv @ (
        #     delta_target * ease(self.sim_time % switch_duration, 0.0, 1.0, switch_duration)
        # )
        # # prefer movement of the first joint to rotate the entire arm towards the goal first
        # delta_q[1:] *= 0.1

        control_strength = ease(
            self.sim_time % switch_duration, 0.0, 0.1, switch_duration
        )

        target_joint_q = wp.array(
            q + delta_q * control_strength,
            device=self.device,
        )
        if True:
            state.joint_q = target_joint_q
            state.joint_qd = wp.array(delta_q, device=self.device)
        else:
            wp.sim.eval_fk(self.model, state.joint_q, state.joint_qd, None, next_state)
            next_state.joint_q = state.joint_q
            next_state.joint_qd = state.joint_qd
            torques = self.control.joint_act

            target_ke = np.zeros(self.model.joint_coord_count, dtype=np.float32)
            target_ke[:7] = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]

            target_kd = np.zeros(self.model.joint_dof_count, dtype=np.float32)
            target_kd[:7] = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]

            generate_pd_control(
                self.model,
                torques,
                state.joint_q,
                state.joint_qd,
                target_q=target_joint_q,
                target_ke=wp.array(target_ke, device=self.device),
                target_kd=wp.array(target_kd, device=self.device),
            )

    def custom_render(self, render_state, renderer):
        if self.show_target_point:
            # renderer.render_points(
            #     "targets", [self.target[:3]], radius=0.025, colors=(1.0, 0.0, 0.0)
            # )
            rot = wp.quat_between_vectors(
                wp.vec3(0.0, 1.0, 0.0), wp.vec3(*self.target[3:6])
            )
            # renderer.render_arrow(
            #     "target",
            #     pos=self.target[:3],
            #     rot=rot,
            #     base_radius=0.01,
            #     base_height=0.07,
            #     cap_height=0.02,
            #     color=(1.0, 0.0, 0.0),
            # )
            for j, target in enumerate(self.targets):
                target_tf = wp.transform(*target[:7])
                for i in range(3):
                    ax = np.zeros(3, dtype=np.float32)
                    ax[i] = 1.0
                    if self.render_mode == RenderMode.OPENGL:
                        self.renderer.render_arrow(
                            f"target_{i}{j}",
                            pos=target_tf.p,
                            rot=target_tf.q,
                            base_radius=0.01,
                            base_height=0.07,
                            cap_height=0.02,
                            color=ax,
                            up_axis=i,
                        )
                    else:
                        self.renderer.render_capsule(
                            f"target_{i}{j}",
                            pos=target_tf.p + ax * 0.07,
                            rot=target_tf.q,
                            radius=0.01,
                            half_height=0.07,
                            color=(float(ax[0]), float(ax[1]), float(ax[2])),
                            # up_axis=i,
                        )

        body_q = self.state.body_q.numpy()
        ee_tf = wp.transform(*body_q[self.endeffector_id]) * self.endeffector_offset
        for i in range(3):
            ax = np.zeros(3, dtype=np.float32)
            ax[i] = 1.0
            if self.render_mode == RenderMode.OPENGL:
                self.renderer.render_arrow(
                    f"ee_{i}",
                    pos=ee_tf.p,
                    rot=ee_tf.q,
                    base_radius=0.01,
                    base_height=0.07,
                    cap_height=0.02,
                    color=ax,
                    up_axis=i,
                )
            else:
                self.renderer.render_capsule(
                    f"ee_{i}",
                    pos=ee_tf.p + ax * 0.07,
                    rot=ee_tf.q,
                    radius=0.01,
                    half_height=0.07,
                    color=(float(ax[0]), float(ax[1]), float(ax[2])),
                    # up_axis=i,
                )


if __name__ == "__main__":
    run_env(FrankaPandaEnvironment)
