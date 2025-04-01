# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Joint targets environment
#
# Shows how to set up rigid body articulations with joint targets defined
# on the position and velocity level.
#
###########################################################################

import warp as wp
import warp.sim

from environment import Environment, run_env, IntegratorType


class JointTargetEnvironment(Environment):
    sim_name = "env_joint_targets"
    env_offset = (2.5, 0.0, 2.5)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=10.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 3

    joint_attach_ke: float = 1e5
    joint_attach_kd: float = 1e3

    use_graph_capture = True
    use_tiled_rendering = True

    num_envs = 1

    activate_ground_plane = False

    # integrator_type = IntegratorType.XPBD
    # integrator_type = IntegratorType.EULER
    integrator_type = IntegratorType.FEATHERSTONE

    def create_articulation(self, builder):
        # create a "rail" for the prismatic joint
        shape1 = builder.add_shape_capsule(body=-1, radius=0.1, half_height=5.0, up_axis=0)
        # create a sphere to slide on this rail
        prismatic_body = builder.add_body()
        shape2 = builder.add_shape_sphere(body=prismatic_body, radius=0.5)
        builder.add_joint_prismatic(
            parent=-1, child=prismatic_body,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(1., 0., 0.),
            mode=wp.sim.JOINT_MODE_TARGET_POSITION,
            limit_lower=-5.0, limit_upper=5.0,
            limit_ke=1e5, limit_kd=1e3,
            target=3.1415,  # target position
            target_ke=1e4,  # target stiffness
            target_kd=1e3,  # target damping
        )
        # make sure we generate no contact between both shapes (the sphere will get stuck otherwise)
        builder.shape_collision_filter_pairs.add((shape1, shape2))

        # add a rotational degree of freedom
        revolute_body = builder.add_body()
        shape3 = builder.add_shape_box(body=revolute_body, hx=0.25, hy=0.25, hz=1.5)
        builder.add_joint_revolute(
            parent=prismatic_body, child=revolute_body,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0., 0., 1.),
            mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
            limit_lower=-1e6,
            limit_upper=1e6,
            target=5.0, target_ke=1e3)
        # prevent contacts with the rail capsule again
        builder.shape_collision_filter_pairs.add((shape1, shape3))


if __name__ == "__main__":
    run_env(JointTargetEnvironment)
