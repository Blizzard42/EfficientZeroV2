# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Ball drop environment
###########################################################################

import warp as wp
import warp.sim

from warp_sim_envs.envs import Environment, run_env, IntegratorType


class BallDropEnvironment(Environment):
    sim_name = "env_ball_drop"

    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 4
    sim_substeps_xpbd = 1

    integrator_type = IntegratorType.FEATHERSTONE
    # integrator_type = IntegratorType.EULER

    activate_ground_plane = True

    def create_articulation(self, builder):
        builder.default_shape_ke = 1e4
        builder.default_shape_kd = 1e2
        b = builder.add_body(name="ball")
        builder.add_shape_sphere(
            b, radius=0.15, density=1000.0,
        )
        builder.add_joint_free(child=b)
        builder.joint_q[1] = 0.5

if __name__ == "__main__":
    run_env(BallDropEnvironment)
