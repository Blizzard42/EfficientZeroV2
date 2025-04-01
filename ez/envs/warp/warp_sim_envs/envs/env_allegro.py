# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Allegro
#
# Shows how to set up a simulation of a rigid-body Allegro hand articulation
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os

import numpy as np
import warp as wp
import warp.sim

from environment import Environment, run_env, IntegratorType


class AllegroEnvironment(Environment):
    sim_name = "example_sim_allegro"
    episode_duration = 8.0

    sim_substeps_euler = 60
    sim_substeps_xpbd = 8
    sim_substeps_featherstone = 30

    rigid_contact_margin = 0.001
    rigid_mesh_contact_max = 100

    num_envs = 4
    # integrator_type = IntegratorType.FEATHERSTONE
    # integrator_type = IntegratorType.EULER

    scale = 1.0

    show_joints = False

    xpbd_settings = dict(
        iterations=2,
        joint_linear_relaxation=1.0,
        joint_angular_relaxation=0.45,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    use_tiled_rendering = False

    edge_sdf_iter = 20

    # use_graph_capture = False

    env_offset = (0.5 * scale, 0.0, 0.5 * scale)
    opengl_render_settings = dict(scaling=5.0 / scale, draw_axis=False)
    usd_render_settings = dict(scaling=200.0 / scale)

    def load_mesh(self, filename, use_meshio=False):
        if use_meshio:
            import meshio

            m = meshio.read(filename)
            mesh_points = np.array(m.points)
            mesh_indices = np.array(m.cells[0].data, dtype=np.int32).flatten()
        else:
            import openmesh

            m = openmesh.read_trimesh(filename)
            mesh_points = np.array(m.points())
            mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32).flatten()
        return wp.sim.Mesh(mesh_points, mesh_indices)

    def create_articulation(self, builder):
        self.ke = 5e4
        self.kd = 1e3
        self.kf = 5e6
        self.mu = 0.5

        if self.integrator_type == IntegratorType.XPBD:
            density_hand = 1e3
            density_cube = 1e2
        else:
            density_hand = 5e4
            density_cube = 1e5

        builder.set_ground_plane(
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            mu=self.mu,
        )

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/kuka_allegro_description/allegro.urdf"),
            builder,
            xform=wp.transform(
                np.array((0.0, 0.3, 0.0)) * self.scale, wp.quat_rpy(-np.pi / 2, np.pi * 0.75, np.pi / 2)
            ),
            floating=False,
            base_joint="rx, ry, rz",
            density=density_hand,
            armature=0.05,
            scale=self.scale,
            stiffness=1000.0,
            damping=0.0,
            contact_ke=self.ke * 5.0,
            contact_kd=self.kd * 5.0,
            contact_kf=self.kf * 5.0,
            contact_mu=self.mu,
            contact_thickness=0.001,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
        )

        # for mesh in builder.shape_geo_src:
        #     if isinstance(mesh, wp.sim.Mesh):
        #         mesh.remesh(visualize=False)

        # apply damping to base DOFs
        # for i in range(3):
        #     builder.joint_target_kd[i] = 100.0

        # ensure all joint positions are within limits
        offset = 3
        for i in range(offset, 16 + offset):
            builder.joint_q[i] = 0.5 * (builder.joint_limit_lower[i] + builder.joint_limit_upper[i])
            builder.joint_act[i] = builder.joint_q[i]
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_TARGET_POSITION
            builder.joint_target_ke[i] = 5000.0
            builder.joint_target_kd[i] = 1.0

        cube_positions = (
            np.array(
                [
                    (-0.1, 0.5, 0.0),
                    (0.0, 0.05, 0.05),
                    (0.01, 0.15, 0.03),
                    (0.01, 0.05, 0.13),
                ]
            )
            * self.scale
        )
        # object_shape = self.load_mesh(os.path.join(os.path.dirname(__file__), "../assets/icosphere.obj"))
        scale = 4e-2 * self.scale
        for pos in cube_positions:
            builder.add_articulation()
            b = builder.add_body()
            # b = builder.add_body(wp.transform(pos, wp.quat_identity()))
            # builder.add_shape_mesh(mesh=object_shape, body=b, density=1e3, scale=(scale, scale, scale))
            builder.add_shape_box(
                body=b,
                density=density_cube,
                hx=scale,
                hy=scale,
                hz=scale,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=self.mu,
            )
            builder.add_joint_free(b)
            builder.joint_q[-7:-4] = pos

        # builder.plot_articulation()
        builder.collapse_fixed_joints()
        # builder.plot_articulation()


if __name__ == "__main__":
    run_env(AllegroEnvironment)
