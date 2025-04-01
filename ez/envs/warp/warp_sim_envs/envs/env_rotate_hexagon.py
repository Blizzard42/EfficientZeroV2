import os
import math

import warp as wp
import warp.render
import warp.sim
import numpy as np

import warp.sim.render
from warp_sim_envs.envs import Environment, run_env, IntegratorType, QueryTargetObjects

TERMINATE_ERROR_REVOLUTION = 3 # Terminate if error more than this revolutions from target
RESET_JOINT_ONE_SIDE_RANGE_MIN = 1/4
RESET_JOINT_ONE_SIDE_RANGE_MAX = TERMINATE_ERROR_REVOLUTION / 4

IMPORT_SCALE = 10.0

# cost = (Error)^2 * scale + bias
ERROR_SCALE = 1.0 #((180.0 / 5) / math.pi) ** 2
ERROR_BIAS = - (10.0 * math.pi / 180.0) ** 2
# 1 unit of error at every ~ degrees pre squared
# 0 error at 1 degrees

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


@wp.kernel(enable_backward=False)
def resample_targets_kernel(
    resample: wp.array(dtype=wp.bool),
    seeds: wp.array(dtype=int),
    random_reset: bool,
    num_envs: int,
    env_offsets: wp.array(dtype=wp.vec3),
    point_height_scale: float,
    point_height_offset: float,
    # outputs
    target_angle: wp.array(dtype=wp.float32),
    target_angle_points: wp.array(dtype=wp.vec3)
):
    env_id = wp.tid()
    seed = seeds[env_id]
    wp.atomic_add(seeds, env_id, num_envs)

    if resample:
        if not resample[env_id]:
            return

    if random_reset:
        random_state = wp.rand_init(seed, env_id)
        target_angle[env_id] = wp.randf(random_state, -wp.PI, wp.PI)

        if target_angle_points:
            target_angle_points[env_id] = wp.vec3(
                0.07 * IMPORT_SCALE,
                point_height_scale * target_angle[env_id] + point_height_offset,
                -0.03 * IMPORT_SCALE
            ) + env_offsets[env_id]


@wp.kernel(enable_backward=False)
def reset_rotate_hexagon(
    reset: wp.array(dtype=wp.bool),
    seeds: wp.array(dtype=int),
    random_reset: bool,
    num_envs: int,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    hex_joint_index: int,
    hex_joint_d_index: int,
    joints_limit_lower: wp.array(dtype=wp.float32),
    joints_limit_upper: wp.array(dtype=wp.float32),
    target_joint_q: wp.array(dtype=wp.float32),
    default_joint_q: wp.array(dtype=wp.float32),
    default_joint_qd: wp.array(dtype=wp.float32),
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    
    env_id = wp.tid()
    seed = seeds[env_id]
    wp.atomic_add(seeds, env_id, num_envs)

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
            joint_qd[env_id * dof_qd_per_env + i] = 0.01 * wp.randf(random_state, -wp.PI, wp.PI)

        # Target joint only
        
        deviation_magnitude = 2.0 * wp.PI * wp.randf(random_state, RESET_JOINT_ONE_SIDE_RANGE_MIN, RESET_JOINT_ONE_SIDE_RANGE_MAX)
        deviation_direction = wp.randf(random_state, -1.0, 1.0)

        joint_q[env_id * dof_q_per_env + hex_joint_index] = target_joint_q[env_id] + wp.sign(deviation_direction) * deviation_magnitude 
        joint_qd[env_id * dof_qd_per_env + hex_joint_d_index] = 0.0 # No rotation for target -> If claw doesnt do anything, it should reach the goal on its own
    else:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q[i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd[i]

@wp.kernel
def rotate_hexagon_target_cost_termination(
    dof_q_per_env: int,
    hex_joint_index: int,
    joint_q: wp.array(dtype=wp.float32),
    target_joint_q: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool)
):
    env_id = wp.tid()

    angle_error = target_joint_q[env_id] - joint_q[env_id * dof_q_per_env + hex_joint_index]

    c = ERROR_SCALE * (angle_error * angle_error) + ERROR_BIAS

    wp.atomic_add(cost, env_id, c)

    if wp.abs(angle_error) > float(TERMINATE_ERROR_REVOLUTION) * (2.0 * wp.PI):
        terminated[env_id] = True
        wp.atomic_add(cost, env_id, 1000.0)


@wp.kernel
def compute_observations_rotate_hexagon(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    hex_joint_index: int,
    hex_joint_d_index: int,
    target_angle: wp.array(dtype=wp.float32),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    # outputs
    obs: wp.array(dtype=float, ndim=2)
):
    env_id = wp.tid()

    # Angle obs
    query_offset = int(0)
    assign_offset = 0
    for i in range(dof_q_per_env - 1):
        if i == hex_joint_index:
            query_offset = query_offset + 1
        obs[env_id, i] = joint_q[env_id * dof_q_per_env + i + query_offset]

    assign_offset = dof_q_per_env - 1

    # Angular Velocities
    for i in range(dof_qd_per_env):
        obs[env_id, i + assign_offset] = joint_qd[env_id * dof_qd_per_env + i]

    assign_offset = assign_offset + dof_qd_per_env

    # Set vector
    obs[env_id, assign_offset] = wp.cos(joint_q[env_id * dof_q_per_env + hex_joint_index])
    obs[env_id, assign_offset + 1] = wp.sin(joint_q[env_id * dof_q_per_env + hex_joint_index])
    assign_offset = assign_offset + 2

    # obs[env_id, assign_offset] = joint_qd[env_id * dof_qd_per_env + hex_joint_d_index]
    # assign_offset = assign_offset + 1

    # Set dAngle
    obs[env_id, assign_offset] = target_angle[env_id] - joint_q[env_id * dof_q_per_env + hex_joint_index]

@wp.kernel
def calculate_progress_visuals(
    dof_q_per_env: int,
    hex_joint_index: int,
    env_offsets: wp.array(dtype=wp.vec3),
    joint_q: wp.array(dtype=wp.float32),
    point_height_scale: float,
    point_height_offset: float,
    # target_angles: wp.array(dtype=wp.float32),
    # outputs
    # target_angle_points: wp.array(dtype=wp.vec3),
    current_angle_points: wp.array(dtype=wp.vec3),
    heading_points: wp.array(dtype=wp.vec3)
):
    env_id = wp.tid()

    current_angle = joint_q[env_id * dof_q_per_env + hex_joint_index]

    current_angle_points[env_id] = wp.vec3(
        0.07 * IMPORT_SCALE, 
        point_height_scale * current_angle + point_height_offset,
        0.03 * IMPORT_SCALE
    ) + env_offsets[env_id]

    heading_points[env_id] = wp.vec3(
        (0.03 * wp.cos(current_angle) + 0.07) * IMPORT_SCALE,
        0.07 * IMPORT_SCALE,
        (0.03 * wp.sin(current_angle)) * IMPORT_SCALE
    ) + env_offsets[env_id]

@wp.kernel
def query_hex_body_q(
    body_q: wp.array(dtype=wp.transform),
    bodies_per_env: int,
    hex_body_offset: int,
    env_offsets: wp.array(dtype=wp.vec3),
    # outputs
    hex_body_q_buffer: wp.array(dtype=wp.float32, ndim=2)
):
    env_id = wp.tid()

    hex_body_q = body_q[env_id * bodies_per_env + hex_body_offset]
    hex_body_translation = wp.transform_get_translation(hex_body_q)
    hex_body_orientation = wp.transform_get_rotation(hex_body_q)

    for i in range(3):
        hex_body_q_buffer[env_id, i] = hex_body_translation[i] - env_offsets[env_id][i]

    for i in range(4):
        hex_body_q_buffer[env_id, 3 + i] = hex_body_orientation[i]

class ClawRotateHexEnvironment(Environment, QueryTargetObjects):
    robot_name = 'ClawRotateHex'
    sim_name = "env_rotate_hexagon"
    env_offset = (2.0, 0.0, 2.5)
    opengl_render_settings = dict(scaling=1.0, axis_scale=0.2)
    usd_render_settings = dict(scaling=10.0)

    sim_substeps_euler = 16
    sim_substeps_featherstone = 4
    sim_substeps_xpbd = 4

    # num_rigid_contacts_per_env = 0

    activate_ground_plane = False

    integrator_type = IntegratorType.FEATHERSTONE
    # featherstone_update_mass_matrix_once_per_step = True

    show_joints = True

    controllable_dofs = [0, 1, 2, 3]
    control_gains = [50.0] * 4
    control_limits = [(-1.0, 1.0)] * 4

    # For testing correctness on simpler controls
    # controllable_dofs = [4]
    # control_gains = [10.0]
    # control_limits = [(-1.0, 1.0)]

    # use_graph_capture = False

    hex_joint_index = 0

    def __init__(
        self,
        seed=42,
        random_reset=True,
        **kwargs,
    ):
        self.seed = seed
        self.random_reset = random_reset
        super().__init__(**kwargs)

        self.seed_wp = wp.array(self.num_envs * [seed], dtype=int)
        self.target_angles = wp.zeros(self.num_envs, dtype=wp.float32)
        self.target_angles_points = wp.zeros(self.num_envs, dtype=wp.vec3)
        self.current_angle_points = wp.zeros(self.num_envs, dtype=wp.vec3)
        self.current_heading_points = wp.zeros(self.num_envs, dtype=wp.vec3)
        self.resample_targets(force_resample=True)
        # print("FORCE")
        # self.reset_envs()

        # print(f'{self.model.joint_q.numpy()=}')
        # print(f'{self.model.joint_limit_lower.numpy()=} {self.model.joint_limit_upper.numpy()=}')

    def create_articulation(self, builder: warp.sim.ModelBuilder):
        warp.sim.parse_urdf(
            os.path.join(
                os.path.dirname(__file__),
                "assets/rotate_hexagon/rotate_hexagon.urdf",
            ),
            builder,
            scale=IMPORT_SCALE,
            floating=False,
            # stiffness=85.0,
            stiffness=0.0,
            # damping=2.0,
            damping=10.0,
            armature=0.06,
            contact_ke=2.0e4,
            contact_kd=5.0e2,
            contact_kf=1.0e4,
            contact_mu=0.75,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            # enable_self_collisions=False,
            enable_self_collisions=True,
            collapse_fixed_joints=True,
            # ignore_visuals=not self.load_visuals,
        )

        # print(f'{builder.body_name=}')
        # print(f'{builder.joint_name=} {builder.joint_type=} {builder.joint_axis_mode=} {[str(j) for j in builder.joint_axis]=}')
        # print(f'{builder.joint_limit_lower=} {builder.joint_limit_upper=}')

        HEX_JOINT_NAME = "hex_joint"
        joint_name_index = builder.joint_name.index(HEX_JOINT_NAME)
        self.hex_joint_index = builder.joint_q_start[joint_name_index]
        self.hex_joint_d_index = builder.joint_qd_start[joint_name_index]

        joint_names = ['claw1_mid_joint', 'claw1_end_joint', 'claw2_mid_joint', 'claw2_end_joint', 'hex_joint']
        joint_names_index_map = { name : i for i, name in enumerate(joint_names)}
        joint_reset_limit_lower = [0.0, 0.0, 0.0, 0.0, -math.pi]
        joint_reset_limit_upper = [np.deg2rad(25.0), np.deg2rad(30.0), np.deg2rad(25.0), np.deg2rad(30.0), math.pi]

        joint_q_reset_limit_lower = np.zeros(len(joint_names), dtype=np.float32)
        joint_q_reset_limit_upper = np.zeros(len(joint_names), dtype=np.float32)

        for i, joint_name in enumerate(builder.joint_name):
            joint_q_reset_limit_lower[i] = joint_reset_limit_lower[joint_names_index_map[joint_name]]
            joint_q_reset_limit_upper[i] = joint_reset_limit_upper[joint_names_index_map[joint_name]]

            builder.joint_q[i] = np.random.uniform(low=joint_q_reset_limit_lower[i], high=joint_q_reset_limit_upper[i])
            builder.joint_qd[i] = 0
        
        self.joint_q_reset_limit_lower = wp.array(joint_q_reset_limit_lower, dtype=wp.float32)
        self.joint_q_reset_limit_upper = wp.array(joint_q_reset_limit_upper, dtype=wp.float32)

        hex_body_name = 'hex_link'
        self.hex_body_q_index = builder.body_name.index(hex_body_name)
        self.target_body_dict = { 'hex': self.hex_body_q_index }

        # ['claw2_mid_joint', 'claw2_end_joint', 'claw1_mid_joint', 'claw1_end_joint', 'hex_joint']e

        # for i in range(len(joint_names)):
        #     builder.joint_q[i] = joint_reset_limit_upper[i]
        # builder.joint_qd =

        # controls = -np.array([10.0] * self.control_dim, dtype=np.float32)
        # controls[-1] = 5.0
        # self.controls_wp = wp.array(controls, dtype=wp.float32)

    def resample_targets(self, env_ids = None, force_resample = False):
        wp.launch(
            resample_targets_kernel,
            dim=self.num_envs,
            inputs = [
                env_ids,
                self.seed_wp,
                self.random_reset or force_resample,
                self.num_envs,
                self.env_offsets_wp,
                self.render_progress_height_scale,
                self.render_progress_offset,
            ],
            outputs = [
                self.target_angles,
                self.target_angles_points
            ],
            device=self.device
        )

        # self.seed += self.num_envs


    def reset_envs(self, env_ids = None, state=None):
        if state is None:
            state = self.state

        # print(f'Pre {state.joint_q.numpy()=}')

        self.resample_targets(env_ids)

        wp.launch(
            reset_rotate_hexagon,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed_wp,
                self.random_reset,
                self.num_envs,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.hex_joint_index,
                self.hex_joint_d_index,
                self.joint_q_reset_limit_lower,
                self.joint_q_reset_limit_upper,
                self.target_angles,
                self.model.joint_q,
                self.model.joint_qd
            ],
            outputs=[
                state.joint_q,
                state.joint_qd,
            ],
            device=self.device,
        )
        # self.seed += self.num_envs
        
        # print(f'Post {state.joint_q.numpy()=}')
        # print(f'{self.joint_q_reset_limit_lower.numpy()=} {self.joint_q_reset_limit_upper.numpy()=}')

        # # update maximal coordinates after reset
        # if env_ids is None or env_ids.numpy().any():
        #     if self.uses_generalized_coordinates:
        # wp.sim.eval_fk(self.model, state.joint_q, state.joint_qd, None, state)
    
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
            rotate_hexagon_target_cost_termination,
            dim=self.num_envs,
            inputs=[
                self.dof_q_per_env,
                self.hex_joint_index,
                state.joint_q,
                self.target_angles
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
        return (self.dof_q_per_env-1) + self.dof_qd_per_env + 2 + 1
        # return 2
    
    def compute_observations(
        self, 
        state: wp.sim.State, 
        control: wp.sim.Control, 
        observations: wp.array, 
        step: int, 
        horizon_length: int
    ):
        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)

        wp.launch(
            compute_observations_rotate_hexagon,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.hex_joint_index,
                self.hex_joint_d_index,
                self.target_angles,
                self.dof_q_per_env,
                self.dof_qd_per_env
            ],
            outputs=[
                observations
            ],
            device=self.device
        )

        # print(f'{observations.numpy()=}')
        
    def get_target_keys(self):
        return list(self.target_body_dict.keys())

    def query_target_body_q(self, state: warp.sim.State, key: str, buffer: wp.array):

        warp.sim.eval_fk(self.model, state.joint_q, state.joint_qd, None, state)

        wp.launch(
            query_hex_body_q,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                self.bodies_per_env,
                # self.hex_body_q_index,
                self.target_body_dict[key],
                self.env_offsets_wp
            ],
            outputs=[
                buffer
            ],
            device=self.device
        )

    def print_state(
        self,
        state: warp.sim.State,
        next_state: warp.sim.State,
        control: warp.sim.Control,
        eval_collisions: bool = True,
    ):
        
        print(f'{self.hex_joint_index=} {state.joint_q.numpy()=}')

    render_progress = True
    render_progress_height_scale = 0.5 / math.pi
    render_progress_offset = 1.5
    render_progress_radius = 0.01 * IMPORT_SCALE
    def custom_render(self, render_state: warp.sim.State, renderer: warp.render.OpenGLRenderer):
        
        # Render Box of height self.target_angles
        # Render Box of height self.current_angles

        wp.launch(
            calculate_progress_visuals,
            dim=self.num_envs,
            inputs = [
                self.dof_q_per_env,
                self.hex_joint_index,
                self.env_offsets_wp,
                render_state.joint_q,
                self.render_progress_height_scale,
                self.render_progress_offset
            ],
            outputs = [
                self.current_angle_points,
                self.current_heading_points
            ],
            device=self.device
        )

        renderer.render_points(
            "targets", self.target_angles_points, radius=self.render_progress_radius, colors=(0.0, 1.0, 0.0)
        )

        renderer.render_points(
            "current", self.current_angle_points, radius=self.render_progress_radius, colors=(1.0, 0.0, 0.0)
        )

        renderer.render_points(
            "heading", self.current_heading_points, radius=self.render_progress_radius, colors=(0.0, 0.0, 1.0)
        )

        # print(f'{self.target_angles.numpy()[0]=} {self.target_angles_points.numpy()[0]=}')
        # print(f'{self.target_angles_points.numpy()[0:4]=}')
        
        pass


def apply_force(self: ClawRotateHexEnvironment, control, **_):
    
    wp.launch(
        apply_controls,
        dim=self.num_envs,
        inputs=[
            self.control_dim,
            self.controls_wp
        ],
        outputs=[
            control.joint_act
        ],
        device=self.device,
    )

    print(f'{control.joint_act.numpy()=}')


if __name__ == "__main__":
    # ClawRotateHexEnvironment.before_step = apply_force
    # ClawRotateHexEnvironment.after_step = ClawRotateHexEnvironment.print_state
    run_env(ClawRotateHexEnvironment)