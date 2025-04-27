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

import math
import numpy as np

import warp as wp
import warp.sim

from warp_sim_envs.envs import Environment, run_env, IntegratorType, QueryTargetObjects
from warp_sim_envs.utils import rotation_distance_function

# Kernel Constants
CUBE_RESET_MIN_ANGLE_DEVIATION = 60.0 * math.pi / 180.0
CUBE_RESET_MAX_ANGLE_DEVIATION = math.pi

@wp.kernel
def initialize_identity_quat(
    quat_array: wp.array(dtype=wp.quat)
):
    env_id = wp.tid()
    quat_array[env_id] = wp.quat_identity()

@wp.kernel(enable_backward=False)
def apply_controls(
    num_dofs: int,
    controls: wp.array(dtype=wp.float32),
    # outputs
    joint_act: wp.array(dtype=wp.float32)
):
    env_id = wp.tid()
    
    for i in range(num_dofs):
        joint_act[env_id * num_dofs + i] = controls[i]

# Resample Target Cube Orientation
@wp.kernel(enable_backward=False)
def resample_targets_kernel(
    resample: wp.array(dtype=wp.bool),
    seeds: wp.array(dtype=int),
    random_reset: bool,
    num_envs: int,
    env_offsets: wp.array(dtype=wp.vec3),
    points_display_distance: float,
    display_offset: wp.vec3,
    # outputs
    target_orientations: wp.array(dtype=wp.quat),
    target_orientations_points_x: wp.array(dtype=wp.vec3),
    target_orientations_points_y: wp.array(dtype=wp.vec3),
    target_orientations_points_z: wp.array(dtype=wp.vec3),
    target_orientations_points_center: wp.array(dtype=wp.vec3),
):
    env_id = wp.tid()
    seed = seeds[env_id]
    seeds[env_id] += num_envs

    if resample:
        if not resample[env_id]:
            return

    if random_reset:
        random_state = wp.rand_init(seed, env_id)

        random_direction = wp.sample_unit_sphere_surface(random_state)
        # random_direction = wp.vec3(wp.randf(random_state, -1.0, 1.0), wp.randf(random_state, -1.0, 1.0), wp.randf(random_state, -1.0, 1.0))
        # random_direction_norm = wp.length(random_direction)
        # random_direction_unit = random_direction / (random_direction_norm + 1e-5)

        target_orientations[env_id] = wp.quat_from_axis_angle(random_direction, wp.randf(random_state, 0.0, wp.PI))

        frame_matrix = wp.quat_to_matrix(target_orientations[env_id])

        if target_orientations_points_x:
            target_orientations_points_x[env_id] = wp.vec3(
                points_display_distance * frame_matrix[0, 0],
                points_display_distance * frame_matrix[1, 0],
                points_display_distance * frame_matrix[2, 0],
            ) + env_offsets[env_id] + display_offset

        if target_orientations_points_y:
            target_orientations_points_y[env_id] = wp.vec3(
                points_display_distance * frame_matrix[0, 1],
                points_display_distance * frame_matrix[1, 1],
                points_display_distance * frame_matrix[2, 1],
            ) + env_offsets[env_id] + display_offset

        if target_orientations_points_z:
            target_orientations_points_z[env_id] = wp.vec3(
                points_display_distance * frame_matrix[0, 2],
                points_display_distance * frame_matrix[1, 2],
                points_display_distance * frame_matrix[2, 2],
            ) + env_offsets[env_id] + display_offset

        if target_orientations_points_center:
            target_orientations_points_center[env_id] = env_offsets[env_id] + display_offset

# Reset environments
@wp.kernel(enable_backward=False)
def reset_rotate_cube(
    reset: wp.array(dtype=wp.bool),
    seeds: wp.array(dtype=int),
    random_reset: bool,
    num_envs: int,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    cube_joint_index_start: int,
    cube_joint_d_index_start: int,
    joints_limit_lower: wp.array(dtype=wp.float32),
    joints_limit_upper: wp.array(dtype=wp.float32),
    
    default_joint_q: wp.array(dtype=wp.float32),
    default_joint_qd: wp.array(dtype=wp.float32),
    
    cube_target_orientations: wp.array(dtype=wp.quat),
    cube_position_deviation_min: wp.vec3,
    cube_position_deviation_max: wp.vec3,

    env_offsets: wp.array(dtype=wp.vec3),
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    
    env_id = wp.tid()
    seed = seeds[env_id]
    seeds[env_id] += num_envs

    if reset:
        if not reset[env_id]:
            return
    
    if random_reset:
        random_state = wp.rand_init(seed, env_id)
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = wp.randf(
                random_state, 
                joints_limit_lower[i],
                joints_limit_upper[i]
                ) 
            
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = 0.0 #0.01 * wp.randf(random_state, -wp.PI, wp.PI)

        # Target cube
        # Set initial position
        position_deviation = wp.cw_mul(wp.sample_unit_cube(random_state), cube_position_deviation_max - cube_position_deviation_min) + cube_position_deviation_min
        for i in range(3):
            joint_q[env_id * dof_q_per_env + cube_joint_index_start + i] = default_joint_q[env_id * dof_q_per_env + cube_joint_index_start + i] + position_deviation[i]

        # Set initial orientation
        deviation_axis = wp.sample_unit_sphere_surface(random_state)
        deviation_angle = wp.randf(random_state, CUBE_RESET_MIN_ANGLE_DEVIATION, CUBE_RESET_MAX_ANGLE_DEVIATION)
        deviation_quat = wp.quat_from_axis_angle(deviation_axis, deviation_angle)

        starting_orientation = cube_target_orientations[env_id] * deviation_quat

        for i in range(4):
            joint_q[env_id * dof_q_per_env + cube_joint_index_start + 3 + i] = starting_orientation[i]
    
    else:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q[i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd[i]

# Cost -> Orientation Deviation
# Terminate if too far out of hand (or too fast?)
ANGLE_REWARD_BIAS = 5.0 * math.pi / 180.0
@wp.kernel
def rotate_cube_target_cost_termination(
    body_q: wp.array(dtype=wp.transform),
    body_q_per_env: int,
    cube_body_index: int,
    target_orientations: wp.array(dtype=wp.quat),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool)
):
    env_id = wp.tid()

    cube_body = body_q[env_id * body_q_per_env + cube_body_index]
    cube_translation = wp.transform_get_translation(cube_body)
    cube_orientation = wp.transform_get_rotation(cube_body)
    # c = rotation_distance_function(cube_orientation, target_orientations[env_id]) - ANGLE_REWARD_BIAS + 0.01 * (cube_translation[1] - 0.1) * (cube_translation[1] - 0.7)
    c = rotation_distance_function(cube_orientation, target_orientations[env_id]) - ANGLE_REWARD_BIAS

    wp.atomic_add(cost, env_id, c)

    if cube_translation[1] < 0.1:
        terminated[env_id] = True

# Compute observation

@wp.kernel
def compute_observations_rotate_cube(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    dof_q_per_env: int,
    dof_qd_per_env: int,

    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_per_env: int,

    env_offset: wp.array(dtype=wp.vec3),

    cube_joint_start_index: int,
    cube_joint_d_start_index: int,
    cube_body_index: int,

    target_orientations: wp.array(dtype=wp.quat),

    # outputs
    obs: wp.array(dtype=float, ndim=2)
):
    env_id = wp.tid()

    # Angle obs
    FREE_JOINT_Q = 7
    query_offset = int(0)
    assign_offset = 0
    for i in range(dof_q_per_env - FREE_JOINT_Q):
        # Skip cube joint index (get from body instead due to inverse kinematics)
        if i == cube_joint_start_index:
            query_offset = query_offset + FREE_JOINT_Q
        obs[env_id, i] = joint_q[env_id * dof_q_per_env + i + query_offset]

    assign_offset = dof_q_per_env - FREE_JOINT_Q

    # Angular Velocities
    FREE_JOINT_QD = 6
    query_offset = int(0)
    for i in range(dof_qd_per_env - FREE_JOINT_QD):
        if i == cube_joint_d_start_index:
            query_offset = query_offset + FREE_JOINT_QD
        obs[env_id, i + assign_offset] = joint_qd[env_id * dof_qd_per_env + i + query_offset]

    assign_offset = assign_offset + dof_qd_per_env - FREE_JOINT_QD

    # Cube Position and Orientation + Velocities
    cube_body = body_q[env_id * body_per_env + cube_body_index]
    cube_translation = wp.transform_get_translation(cube_body)
    cube_orientation = wp.transform_get_rotation(cube_body)
    cube_body_qd = body_qd[env_id * body_per_env + cube_body_index]

    for i in range(3):
        obs[env_id, i + assign_offset] = cube_translation[i] - env_offset[env_id][i]
    assign_offset = assign_offset + 3

    for i in range(4):
        obs[env_id, i + assign_offset] = cube_orientation[i]
    assign_offset = assign_offset + 4

    for i in range(6):
        obs[env_id, i + assign_offset] = cube_body_qd[i]
    assign_offset = assign_offset + 6
    
    # Target Orientation
    target_orientation = target_orientations[env_id]

    for i in range(4):
        obs[env_id, i + assign_offset] = target_orientation[i]
    assign_offset = assign_offset + 4

@wp.kernel
def dummy_observation(
    obserations: wp.array(dtype=wp.float32, ndim=2)
):
    env_id = wp.tid()
    obserations[env_id , 0] = 0.0
    obserations[env_id , 1] = 1.0

@wp.kernel
def query_cube_body_q(
    body_q: wp.array(dtype=wp.transform),
    bodies_per_env: int,
    cube_body_offset: int,
    env_offsets: wp.array(dtype=wp.vec3),
    # outputs
    hex_body_q_buffer: wp.array(dtype=wp.float32, ndim=2)
):
    env_id = wp.tid()

    hex_body_q = body_q[env_id * bodies_per_env + cube_body_offset]
    hex_body_translation = wp.transform_get_translation(hex_body_q)
    hex_body_orientation = wp.transform_get_rotation(hex_body_q)

    for i in range(3):
        hex_body_q_buffer[env_id, i] = hex_body_translation[i] - env_offsets[env_id][i]

    for i in range(4):
        hex_body_q_buffer[env_id, 3 + i] = hex_body_orientation[i]

# See progress using 3 points on cube + 3 target points
@wp.kernel
def calculate_progress_visuals(
    body_q: wp.array(dtype=wp.transform),
    bodies_per_env: int,
    env_offsets: wp.array(dtype=wp.vec3),
    cube_body_index: int,
    points_display_distance: float,
    # target_angles: wp.array(dtype=wp.float32),
    # outputs
    # target_angle_points: wp.array(dtype=wp.vec3),
    current_orientation_points_x: wp.array(dtype=wp.vec3),
    current_orientation_points_y: wp.array(dtype=wp.vec3),
    current_orientation_points_z: wp.array(dtype=wp.vec3),
):
    env_id = wp.tid()

    cube_body = body_q[env_id * bodies_per_env + cube_body_index]
    cube_translation = wp.transform_get_translation(cube_body)
    cube_orientation = wp.transform_get_rotation(cube_body)

    frame_matrix = wp.quat_to_matrix(cube_orientation)

    if current_orientation_points_x:
        current_orientation_points_x[env_id] = wp.vec3(
            points_display_distance * frame_matrix[0, 0],
            points_display_distance * frame_matrix[1, 0],
            points_display_distance * frame_matrix[2, 0],
        ) + cube_translation

    if current_orientation_points_y:
        current_orientation_points_y[env_id] = wp.vec3(
            points_display_distance * frame_matrix[0, 1],
            points_display_distance * frame_matrix[1, 1],
            points_display_distance * frame_matrix[2, 1],
        ) + cube_translation

    if current_orientation_points_z:
        current_orientation_points_z[env_id] = wp.vec3(
            points_display_distance * frame_matrix[0, 2],
            points_display_distance * frame_matrix[1, 2],
            points_display_distance * frame_matrix[2, 2],
        ) + cube_translation


class AllegroRotateCubeEnvironment(Environment, QueryTargetObjects):
    sim_name = "env_allegro_rotate_cube"

    scale = 1.0

    env_offset = (1.0 * scale, 0.0, 1.0 * scale)

    opengl_render_settings = dict(scaling=5.0 / scale, draw_axis=False)
    usd_render_settings = dict(scaling=200.0 / scale)

    sim_substeps_euler = 60
    sim_substeps_xpbd = 8
    sim_substeps_featherstone = 30

    rigid_contact_margin = 0.001
    rigid_mesh_contact_max = 100

    # num_envs = 4

    # integrator_type = IntegratorType.FEATHERSTONE
    # integrator_type = IntegratorType.EULER
    integrator_type = IntegratorType.XPBD

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

    # Cube sim parameters
    cube_position_deviation_min = wp.vec3(-0.02 * scale, 0.0 * scale, -0.02 * scale)
    cube_position_deviation_max = wp.vec3(0.02 * scale, 0.0 * scale, 0.02 * scale)

    # Cube display parameters
    display_sphere_distance = 4e-2 * scale
    display_frame_offset = wp.vec3([-0.1 * scale, 0.4 * scale, 0.15 * scale])
    display_sphere_radius = 0.015 * scale

    # Action parameters
    # controllable_dofs = [i for i in range(19)]
    # control_gains = [10.0] * 3 + [1.0] * (19 - 3)
    # control_limits = [
    #     (-1.0, 1.0),
    #     (-1.0, 1.0),
    #     (-1.0, 1.0),
    #     (0.2793, 1.5707),
    #     (-0.3317, 1.1518),
    #     (-0.2793, 1.7278),
    #     (-0.2793, 1.7627),
    #     (-0.5585, 0.5584),
    #     (-0.2793, 1.7278),
    #     (-0.2793, 1.7278),
    #     (-0.2793, 1.7278),
    #     (-0.5585, 0.5584),
    #     (-0.2793, 1.7278),
    #     (-0.2793, 1.7278),
    #     (-0.2793, 1.7278),
    #     (-0.5585, 0.5584),
    #     (-0.2793, 1.7278),
    #     (-0.2793, 1.7278),
    #     (-0.2793, 1.7278),
    # ]

    controllable_dofs = [i for i in range(19)]
    control_gains = [1.0] * 3 + [1.0] * (19 - 3)
    control_limits = [
        (-math.pi*0.75, math.pi*0.75),
        (-math.pi*0.75, math.pi*0.75),
        (-math.pi*0.75, math.pi*0.75),
        (0.2793, 1.5707),
        (-0.3317, 1.1518),
        (-0.2793, 1.7278),
        (-0.2793, 1.7627),
        (-0.5585, 0.5584),
        (-0.2793, 1.7278),
        (-0.2793, 1.7278),
        (-0.2793, 1.7278),
        (-0.5585, 0.5584),
        (-0.2793, 1.7278),
        (-0.2793, 1.7278),
        (-0.2793, 1.7278),
        (-0.5585, 0.5584),
        (-0.2793, 1.7278),
        (-0.2793, 1.7278),
        (-0.2793, 1.7278),
    ]


    control_limits = [(-1.0, 1.0)] * 19

    def __init__(
        self,
        seed = 42,
        random_reset=True,
        **kwargs
    ):
        self.seed = seed
        self.random_reset = random_reset
        super().__init__(**kwargs)#, use_graph_capture=False)

        self.seed_wp = wp.array(self.num_envs * [self.seed], dtype=int)

        # Set target
        self.target_orientations = wp.zeros(self.num_envs, dtype=wp.quat)
        wp.launch(
            initialize_identity_quat,
            dim=self.num_envs,
            outputs=[self.target_orientations],
            device=self.device
        )

        # Progress Display
        self.target_orientation_points_x =  wp.zeros(self.num_envs, dtype=wp.vec3)
        self.target_orientation_points_y =  wp.zeros(self.num_envs, dtype=wp.vec3)
        self.target_orientation_points_z =  wp.zeros(self.num_envs, dtype=wp.vec3)
        self.target_orientation_points_center = wp.zeros(self.num_envs, dtype=wp.vec3)

        self.current_orientation_points_x =  wp.zeros(self.num_envs, dtype=wp.vec3)
        self.current_orientation_points_y =  wp.zeros(self.num_envs, dtype=wp.vec3)
        self.current_orientation_points_z =  wp.zeros(self.num_envs, dtype=wp.vec3)

        # Reset
        self.reset_envs()
        self.resample_targets(force_resample=True)

        pass


    def create_articulation(self, builder: warp.sim.ModelBuilder):
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

        warp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/kuka_allegro_description/allegro.urdf"),
            builder,
            xform=wp.transform(
                # np.array((0.0, 0.3, 0.0)) * self.scale, wp.quat_rpy(-np.pi / 2, np.pi * 0.75, np.pi / 2)
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

        # Reset limits
        joint_q_reset_limit_lower = np.zeros(19, dtype=np.float32)
        joint_q_reset_limit_upper = np.zeros(19, dtype=np.float32)

        # Controls
        # self.controllable_dofs = []
        # self.control_gains = []
        # self.control_limits = []

        # Base DOF
        # builder.joint_q[0] = -math.pi/2
        for i in range(3):
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_TARGET_POSITION
            builder.joint_target_ke[i] = 20.0
            builder.joint_target_kd[i] = 1.0
            builder.joint_act[i] = builder.joint_q[i]
            # print(f'{i} {builder.joint_q[i]=}')

            # self.controllable_dofs.append(i)
            # self.control_gains.append(10.0)
            # self.control_limits.append((-1.0, 1.0))
            builder.joint_limit_lower[i] = -math.pi * 0.75
            builder.joint_limit_upper[i] = math.pi * 0.75
            

        # ensure all joint positions are within limits
        # Note: Lower -> Open Hand ; Upper -> Close Hand
        offset = 3
        for i in range(offset, 16 + offset):
            builder.joint_q[i] = 0.25 * (builder.joint_limit_lower[i] + builder.joint_limit_upper[i])
            builder.joint_act[i] = builder.joint_q[i]
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_TARGET_POSITION
            builder.joint_target_ke[i] = 20.0#5000.0
            builder.joint_target_kd[i] = 1.0

            joint_limit_range = builder.joint_limit_upper[i] - builder.joint_limit_lower[i]
            joint_q_reset_limit_lower[i] = builder.joint_limit_lower[i]
            joint_q_reset_limit_upper[i] = 0.5 * joint_limit_range + builder.joint_limit_lower[i]

            # self.controllable_dofs.append(i)
            # self.control_gains.append(5.0)
            # self.control_limits.append((-math.pi, math.pi))

        joint_q_reset_limit_lower[:offset] = -10.0 * math.pi / 180.0
        joint_q_reset_limit_upper[:offset] = 10.0 * math.pi / 180.0

        self.joint_q_reset_limit_lower = wp.array(joint_q_reset_limit_lower, dtype=wp.float32)
        self.joint_q_reset_limit_upper = wp.array(joint_q_reset_limit_upper, dtype=wp.float32)

        # cube_initial_position = np.array([-0.1, 0.50, 0.0], dtype=np.float32) * self.scale
        cube_initial_position = np.array([-0.15, 0.50, 0.0], dtype=np.float32) * self.scale
        cube_scale = 4e-2 * self.scale

        # print(f'{builder.joint_name=} {builder.joint_axis_mode=} {builder.joint_limit_lower=} {builder.joint_limit_upper=}')
        # print(f'{builder.joint_axis_start=} {builder.joint_axis_mode=} {builder.joint_axis_count=}')
        # print(f'{builder.joint_target_kd=} {builder.joint_target_ke=} {builder.joint_linear_compliance=} {builder.joint_angular_compliance=}')

        # print("\n".join([str((a,b)) for a,b in zip(builder.joint_limit_lower, builder.joint_limit_upper)]))

        # New cube articulation
        # builder.add_articulation()
        cube_body = builder.add_body(name='play_cube')
        cube_shape = builder.add_shape_box(
            body=cube_body,
            density=density_cube,
            hx=cube_scale,
            hy=cube_scale,
            hz=cube_scale,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            mu=self.mu,
        )

        cube_joint_name = 'joint_free_cube'

        builder.add_joint_free(cube_body, name=cube_joint_name)
        builder.joint_q[-7:-4] = cube_initial_position
        builder.joint_q[-4:] = wp.quat_from_axis_angle(wp.vec3(np.random.uniform(size=3)), np.random.uniform(low=-math.pi,high=math.pi))
    
        # builder.plot_articulation()
        builder.collapse_fixed_joints()
        # builder.plot_articulation()

        self.cube_body_index = builder.body_name.index('play_cube')
        self.cube_joint_start_index = builder.joint_q_start[builder.joint_name.index(cube_joint_name)]
        self.cube_joint_d_start_index = builder.joint_qd_start[builder.joint_name.index(cube_joint_name)]

        self.target_body_dict = { 'cube': self.cube_body_index }

        # controls = np.array(builder.joint_q, dtype=np.float32)
        # controls[0] = 0.0
        # controls[0] = 0.0
        # controls[-1] = 5.0
        # controls[4] = 1.5707
        # self.controls_wp = wp.array(controls, dtype=wp.float32)

    def resample_targets(self, env_ids = None, force_resample=False):
        wp.launch(
            resample_targets_kernel,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed_wp,
                self.random_reset or force_resample,
                self.num_envs,
                self.env_offsets_wp,
                self.display_sphere_distance,
                self.display_frame_offset
            ],
            outputs=[
                self.target_orientations,
                self.target_orientation_points_x,
                self.target_orientation_points_y,
                self.target_orientation_points_z,
                self.target_orientation_points_center
            ],
            device=self.device
        )
    
    def reset_envs(self, env_ids = None, state=None):
        if state is None:
            state = self.state

        # print(f'Pre {state.joint_q.numpy()=}')
        # print(f'Pre {state.joint_q.numpy()=} {state.body_q.numpy()[self.cube_body_index::self.bodies_per_env]=} {self.cube_body_index=} {self.cube_joint_start_index=}')

        # print(f'Pre Targets: {self.target_orientations.numpy()=}')
        self.resample_targets(env_ids)
        # print(f'Post Targets: {self.target_orientations.numpy()=}')

        wp.launch(
            reset_rotate_cube,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed_wp,
                self.random_reset,
                self.num_envs,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.cube_joint_start_index,
                self.cube_joint_d_start_index,
                self.joint_q_reset_limit_lower,
                self.joint_q_reset_limit_upper,
                self.model.joint_q,
                self.model.joint_qd,
                self.target_orientations,
                self.cube_position_deviation_min,
                self.cube_position_deviation_max,
                self.env_offsets_wp
            ],
            outputs=[
                state.joint_q,
                state.joint_qd,
            ],
            device=self.device,
        )

        # print(f'Post {state.joint_q.numpy()=}')
        # print(f'Post {state.joint_q.numpy()=} {state.body_q.numpy()[self.cube_body_index::self.bodies_per_env]=} {self.cube_body_index=} {self.cube_joint_start_index=}')
        # print(f'{self.joint_q_reset_limit_lower.numpy()=} {self.joint_q_reset_limit_upper.numpy()=}')

        # # update maximal coordinates after reset
        # if env_ids is None or env_ids.numpy().any():
        # if self.uses_generalized_coordinates:
        #     wp.sim.eval_fk(self.model, state.joint_q, state.joint_qd, None, state)
        # if env_ids is None or env_ids.numpy().any():
        #     if self.uses_generalized_coordinates:
        #         wp.sim.eval_fk(self.model, state.joint_q, state.joint_qd, None, state)
        if env_ids is not None:
            wp.sim.eval_fk(self.model, state.joint_q, state.joint_qd, env_ids, state)

    def compute_cost_termination(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        wp.launch(
            rotate_cube_target_cost_termination,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                self.bodies_per_env,
                self.cube_body_index,
                self.target_orientations
            ],
            outputs=[
                cost,
                terminated
            ],
            device=self.device
        )

    @property
    def observation_dim(self):
        # Joint q, joint qd, hex heading vector, target deviation angle
        return self.dof_q_per_env + self.dof_qd_per_env + 4
        # return 2
    

    def compute_observations(
        self, 
        state: wp.sim.State, 
        control: wp.sim.Control, 
        observations: wp.array, 
        step: int, 
        horizon_length: int
    ):
        

        # wp.launch(
        #     dummy_observation,
        #     dim=self.num_envs,
        #     outputs=[
        #         observations
        #     ],
        #     device=self.device
        # )

        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)

        wp.launch(
            compute_observations_rotate_cube,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                state.body_q,
                state.body_qd,
                self.bodies_per_env,
                self.env_offsets_wp,
                self.cube_joint_start_index,
                self.cube_joint_d_start_index,
                self.cube_body_index,
                self.target_orientations
            ],
            outputs=[
                observations
            ],
            device=self.device
        )

    def get_target_keys(self):
        return list(self.target_body_dict.keys())

    def query_target_body_q(self, state: warp.sim.State, key: str, buffer: wp.array):

        wp.launch(
            query_cube_body_q,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                self.bodies_per_env,
                # self.cube_body_index,
                self.target_body_dict[key],
                self.env_offsets_wp
            ],
            outputs=[
                buffer
            ],
            device=self.device
        )

    def custom_render(self, render_state: warp.sim.State, renderer: warp.render.OpenGLRenderer):

        wp.launch(
            calculate_progress_visuals,
            dim=self.num_envs,
            inputs = [
                render_state.body_q,
                self.bodies_per_env,
                self.env_offsets_wp,
                self.cube_body_index,
                self.display_sphere_distance
            ],
            outputs = [
                self.current_orientation_points_x,
                self.current_orientation_points_y,
                self.current_orientation_points_z
            ],
            device=self.device
        )

        renderer.render_points(
            "targets_x", self.target_orientation_points_x, radius=self.display_sphere_radius, colors=(1.0, 0.0, 0.0)
        )

        renderer.render_points(
            "targets_y", self.target_orientation_points_y, radius=self.display_sphere_radius, colors=(0.0, 1.0, 0.0)
        )

        renderer.render_points(
            "targets_z", self.target_orientation_points_z, radius=self.display_sphere_radius, colors=(0.0, 0.0, 1.0)
        )

        renderer.render_points(
            "targets_center", self.target_orientation_points_center, radius=self.display_sphere_radius/2, colors=(0.0, 0.0, 0.0)
        )

        renderer.render_points(
            "current_x", self.current_orientation_points_x, radius=self.display_sphere_radius, colors=(1.0, 0.0, 0.0)
        )

        renderer.render_points(
            "current_y", self.current_orientation_points_y, radius=self.display_sphere_radius, colors=(0.0, 1.0, 0.0)
        )

        renderer.render_points(
            "current_z", self.current_orientation_points_z, radius=self.display_sphere_radius, colors=(0.0, 0.0, 1.0)
        )

        # renderer.render_points(
        #     "current", self.current_angle_points, radius=self.render_progress_radius, colors=(1.0, 0.0, 0.0)
        # )

        # renderer.render_points(
        #     "heading", self.current_heading_points, radius=self.render_progress_radius, colors=(0.0, 0.0, 1.0)
        # )
        
        # print(f'{self.target_orientations.numpy()[0]=} {self.target_orientation_points_x.numpy()[0]=}')

        pass

# def apply_force(self: AllegroRotateCubeEnvironment, control, **_):
    
#     wp.launch(
#         apply_controls,
#         dim=self.num_envs,
#         inputs=[
#             self.control_dim,
#             self.controls_wp
#         ],
#         outputs=[
#             control.joint_act
#         ],
#         device=self.device,
#     )

#     print(f'{control.joint_act.numpy()=}')

if __name__ == "__main__":
    # AllegroRotateCubeEnvironment.before_step = apply_force
    run_env(AllegroRotateCubeEnvironment)
