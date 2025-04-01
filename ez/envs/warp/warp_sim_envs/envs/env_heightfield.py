# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Heightfield environment
#
# Demonstrates how to create a heightfield mesh procedurally to be used as
# an uneven ground.
###########################################################################

import os
import math
import warp as wp
import warp.sim

import numpy as np

from warp_sim_envs.envs import Environment, run_env, IntegratorType


class HeightfieldEnvironment(Environment):
    sim_name = "env_heightfield"

    # whether to use rigid bodies or particles for the grains
    use_rigids = True

    env_offset = (20.0, 0.0, 20.0)
    opengl_render_settings = dict(scaling=0.25)
    usd_render_settings = dict(scaling=5.0)

    sim_substeps_euler = 30
    sim_substeps_xpbd = 5

    xpbd_settings = dict(
        iterations=8,
        # enable_restitution=True,
        rigid_contact_con_weighting=False,
        soft_contact_relaxation=0.2,
    )

    num_envs = 1

    activate_ground_plane = False

    def create_articulation(self, builder: wp.sim.ModelBuilder):
        self.ke = 1.0e2
        self.kd = 25.0
        self.kf = 5.0
        self.mu = 0.5
        self.restitution = 0.9

        # sample heights in a height field
        num_x = 100
        num_z = 100
        hy = 0.3
        heights = np.random.uniform(-hy, hy, size=(num_x, num_z))
        # create vertices
        hx = 25.0
        hz = 25.0
        vertices = np.hstack((
            np.repeat(np.linspace(-hx, hx, num_x), num_z).reshape(-1, 1),
            heights.flatten().reshape(-1, 1),
            np.tile(np.linspace(-hz, hz, num_z), num_x).reshape(-1, 1),
        ))
        # create indices
        indices = []
        for i in range(num_x - 1):
            for j in range(num_z - 1):
                # triangle 1
                indices.append(i * num_z + j)
                indices.append(i * num_z + j + 1)
                indices.append((i + 1) * num_z + j)
                # triangle 2
                indices.append(i * num_z + j + 1)
                indices.append((i + 1) * num_z + j + 1)
                indices.append((i + 1) * num_z + j)
        indices = np.array(indices, dtype=np.int32)

        builder.add_shape_mesh(
            body=-1,
            density=0.0,  # make it static
            mesh=wp.sim.Mesh(vertices=vertices, indices=indices, compute_inertia=False),
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            mu=self.mu,
            restitution=self.restitution,
        )

        # create spheres
        height = 3
        width = 10
        depth = 10

        radius = 0.2
        spacing = 0.7
        mass = radius * radius * radius * 0.5 * 100.0
        # spheres
        for i in range(height):
            for j in range(width):
                for k in range(depth):
                    pos = np.array(
                        (
                            (k - (width - 1) / 2.0) * (spacing + radius),
                            i * (spacing + radius) + 22.0,
                            (j - (depth - 1) / 2.0) * (spacing + radius),
                        )
                    )
                    # add jitter
                    pos += np.random.uniform(-spacing, spacing, size=3) * 0.5

                    if self.use_rigids:
                        b = builder.add_body(origin=wp.transform(pos, wp.quat_identity()))

                        builder.add_shape_sphere(
                            pos=(0.0, 0.0, 0.0),
                            radius=radius,
                            body=b,
                            ke=self.ke,
                            kd=self.kd,
                            kf=self.kf,
                            mu=self.mu,
                            restitution=self.restitution,
                        )
                    else:
                        builder.add_particle(pos, vel=wp.vec3(0.0), mass=mass, radius=radius)

        builder.ground = [0.0, 1.0, 0.0, 0.0]

    def before_simulate(self):
        self.model.rigid_contact_rolling_friction = 0.05
        self.model.rigid_contact_torsion_friction = 0.05
        self.model.rigid_contact_margin = 0.05
        self.model.soft_contact_restitution = self.restitution
        # self.model.soft_contact_margin = 0.3


if __name__ == "__main__":
    run_env(HeightfieldEnvironment)
