# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Funnel environment
#
# Pours a stream of balls into a funnel and simulates the dynamics of the
# balls rolling down the funnel. The balls can be represented as either
# rigid bodies or particles.
###########################################################################

import os
import math
import warp as wp
import warp.sim

import numpy as np

from warp_sim_envs.envs import Environment, run_env, IntegratorType


class FunnelEnvironment(Environment):
    sim_name = "env_funnel"

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

    num_envs = 4

    def create_articulation(self, builder: wp.sim.ModelBuilder):
        self.ke = 1.0e2
        self.kd = 25.0
        self.kf = 5.0
        self.mu = 0.5
        self.restitution = 0.9

        builder.set_ground_plane(
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            mu=self.mu,
            restitution=self.restitution,
        )

        if True:
            # funnel
            vertices, faces = warp.sim.load_mesh(
                os.path.join(os.path.dirname(__file__), "assets", "funnel.obj"),
            )
            funnel_mesh = wp.sim.Mesh(vertices, faces)
            builder.add_shape_mesh(
                body=-1,
                mesh=funnel_mesh,
                pos=(0.0, 12.0, 0.0),
                rot=(wp.sin(math.pi / 4), 0.0, 0.0, wp.sin(math.pi / 4)),
                scale=(0.1, 0.1, 0.1),
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=self.mu,
                restitution=self.restitution,
                density=0.0,
                thickness=0.0,
            )
        else:
            builder.add_shape_sphere(
                body=-1,
                radius=3.0,
                pos=(0.0, 12.0, 0.0),
                rot=(wp.sin(math.pi / 4), 0.0, 0.0, wp.sin(math.pi / 4)),
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=self.mu,
                restitution=self.restitution,
                density=0.0,
                thickness=0.01,
            )

        height = 30
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
    run_env(FunnelEnvironment)
