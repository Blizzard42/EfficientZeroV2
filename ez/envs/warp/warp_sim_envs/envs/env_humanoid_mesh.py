# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Humanoid environment
#
# Shows how to set up a simulation of a rigid-body Humanoid articulation based
# on the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import warp as wp
import warp.sim
import warp.examples

from warp_sim_envs.envs import run_env, IntegratorType, HumanoidEnvironment


class HumanoidMeshEnvironment(HumanoidEnvironment):
    num_envs = 100

    env_offset = (5.0, 0.0, 5.0)

    integrator_type = IntegratorType.XPBD

    activate_ground_plane = False

    # XXX set this variable to nonzero to enable mesh contact limiting
    # rigid_mesh_contact_max = 100

    def create_articulation(self, builder):
        super().create_articulation(builder)

        vertices, faces = wp.sim.utils.load_mesh(
            os.path.join(os.path.dirname(__file__), "assets", "bump.obj")
        )
        mesh = wp.sim.Mesh(vertices, faces)

        # add a mesh object
        builder.add_shape_mesh(body=-1, mesh=mesh, pos=(0.0, -0.3, 0.0))


if __name__ == "__main__":
    run_env(HumanoidMeshEnvironment)
