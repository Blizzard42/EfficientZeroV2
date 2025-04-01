# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from enum import Enum
from typing import Tuple, Callable

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from warp.sim.integrator_featherstone import (
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_mass,
)

from warp_sim_envs.utils import (
    eval_kinetic_energy,
    eval_potential_energy,
    generate_pd_control,
    convert_joint_torques_to_body_forces,
    assign_controls,
)


from tqdm import trange

from typing import List


class RenderMode(Enum):
    NONE = "none"
    OPENGL = "opengl"
    USD = "usd"

    def __str__(self):
        return self.value


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    FEATHERSTONE = "featherstone"
    NEURAL = "neural"

    def __str__(self):
        return self.value


def compute_env_offsets(
    num_envs, env_offset=(5.0, 0.0, 5.0), up_axis="Y"
) -> List[np.ndarray]:
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets


@wp.kernel
def assign_joint_q_qd_obs(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    joint_type: wp.array(dtype=int),
    env_offsets: wp.array(dtype=wp.vec3),
    # outputs
    obs: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    # check if articulation is a floating-base system, then subtract env offset
    q_offset = 0
    if joint_type[tid] == wp.sim.JOINT_FREE:
        env_offset = env_offsets[tid]
        for i in range(3):
            obs[tid, i] = joint_q[tid * dof_q_per_env + i] - env_offset[i]
        q_offset = 3
    for i in range(q_offset, dof_q_per_env):
        obs[tid, i] = joint_q[tid * dof_q_per_env + i]
    for i in range(dof_qd_per_env):
        obs[tid, i + dof_q_per_env] = joint_qd[tid * dof_qd_per_env + i]


@wp.kernel(enable_backward=False)
def reset_maximal_coords(
    reset: wp.array(dtype=wp.bool),
    body_q_init: wp.array(dtype=wp.transform),
    body_qd_init: wp.array(dtype=wp.spatial_vector),
    num_bodies_per_env: int,
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    env_id = wp.tid()
    if reset:
        if not reset[env_id]:
            return
    for i in range(num_bodies_per_env):
        j = env_id * num_bodies_per_env + i
        body_q[j] = body_q_init[j]
        body_qd[j] = body_qd_init[j]


@wp.kernel(enable_backward=False)
def reset_generalized_coords(
    reset: wp.array(dtype=wp.bool),
    joint_q_init: wp.array(dtype=wp.float32),
    joint_qd_init: wp.array(dtype=wp.float32),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset:
        if not reset[env_id]:
            return
    # wp.printf("resetting env %i (%i dofs)\n", env_id, dof_q_per_env)
    for i in range(dof_q_per_env):
        j = env_id * dof_q_per_env + i
        joint_q[j] = joint_q_init[j]
    for i in range(dof_qd_per_env):
        j = env_id * dof_qd_per_env + i
        joint_qd[j] = joint_qd_init[j]


class Environment:
    sim_name: str = "Environment"

    frame_dt = 1.0 / 60.0

    episode_duration = 5.0  # seconds
    episode_frames = (
        None  # number of steps per episode, if None, use episode_duration / frame_dt
    )

    # whether to play the simulation indefinitely when using the OpenGL renderer
    continuous_opengl_render: bool = True

    sim_substeps_euler: int = 16
    sim_substeps_featherstone: int = 10
    featherstone_update_mass_matrix_once_per_step: bool = True
    sim_substeps_xpbd: int = 5

    euler_settings = dict()
    featherstone_settings = dict()
    xpbd_settings = dict()

    num_envs = 100
    env_offset = (1.0, 0.0, 1.0)

    render_mode: RenderMode = RenderMode.OPENGL
    opengl_render_settings = dict()
    opengl_tile_render_settings = dict(
        tile_width=128, tile_height=128, move_cam_pos_to_env_offsets=True
    )
    usd_render_settings = dict(scaling=10.0)
    usd_save_as_usda = False

    show_rigid_contact_points = False
    contact_points_radius = 1e-3
    show_joints = False
    # whether OpenGLRenderer should render each environment in a separate tile
    use_tiled_rendering = False

    # whether to apply model.joint_q, joint_qd to bodies before simulating
    eval_fk: bool = True

    use_graph_capture: bool = None

    activate_ground_plane: bool = True

    integrator_type: IntegratorType = IntegratorType.XPBD

    up_axis: str = "Y"
    gravity: float = -9.81

    # stiffness and damping for joint attachment dynamics used by Euler
    joint_attach_ke: float = 32000.0
    joint_attach_kd: float = 50.0

    # maximum number of rigid contact points to generate per mesh
    rigid_mesh_contact_max: int = 0  # (0 = unlimited)

    # distance threshold at which contacts are generated
    rigid_contact_margin: float = 0.05
    # whether to iterate over mesh vertices for box/capsule collision
    rigid_contact_iterate_mesh_vertices: bool = True
    # number of rigid contact points to allocate in the model during self.finalize() per environment
    # if setting is None, the number of worst-case number of contacts will be calculated in self.finalize()
    num_rigid_contacts_per_env: int = None

    # whether to call warp.sim.collide() just once per update
    handle_collisions_once_per_step: bool = False

    # number of search iterations for finding closest contact points between edges and SDF
    edge_sdf_iter: int = 10

    # whether each environment should have its own collision group
    # to avoid collisions between environments
    separate_collision_group_per_env: bool = True

    plot_body_coords: bool = False
    plot_joint_coords: bool = False
    plot_joint_coords_qd: bool = False

    # custom dynamics function to be called instead of the default simulation step
    # signature: custom_dynamics(model, state, next_state, sim_dt, control)
    custom_dynamics: Callable[
        [wp.sim.Model, wp.sim.State, wp.sim.State, float, wp.sim.Control], None
    ] = None

    # control-related definitions, to be updated by derived classes
    controllable_dofs = []
    control_gains = []
    control_limits = []

    def __init__(
        self,
        num_envs: int = None,
        episode_frames: int = None,
        integrator_type: IntegratorType = None,
        render_mode: RenderMode = None,
        env_offset: Tuple[float, float, float] = None,
        device: wp.context.Devicelike = None,
        requires_grad: bool = False,
        profile: bool = False,
        enable_timers: bool = False,
        use_graph_capture: bool = None,
        use_tiled_rendering: bool = None,
        setup_renderer: bool = True,
    ):
        if num_envs is not None:
            self.num_envs = num_envs
        if episode_frames is not None:
            self.episode_frames = episode_frames
        if integrator_type is not None:
            self.integrator_type = integrator_type
        if render_mode is not None:
            self.render_mode = render_mode
        if use_graph_capture is not None:
            self.use_graph_capture = use_graph_capture
        if self.use_graph_capture is None:
            self.use_graph_capture = wp.get_preferred_device().is_cuda
        if use_tiled_rendering is not None:
            self.use_tiled_rendering = use_tiled_rendering
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad
        self.profile = profile
        self.enable_timers = enable_timers

        if self.use_tiled_rendering and self.render_mode == RenderMode.OPENGL:
            # no environment offset when using tiled rendering
            self.env_offset = (0.0, 0.0, 0.0)
        elif env_offset is not None:
            self.env_offset = env_offset

        if isinstance(self.up_axis, str):
            up_vector = np.zeros(3)
            up_vector["xyz".index(self.up_axis.lower())] = 1.0
        else:
            up_vector = self.up_axis
        builder = wp.sim.ModelBuilder(up_vector=up_vector, gravity=self.gravity)
        builder.rigid_mesh_contact_max = self.rigid_mesh_contact_max
        builder.rigid_contact_margin = self.rigid_contact_margin
        builder.num_rigid_contacts_per_env = self.num_rigid_contacts_per_env
        self.env_offsets = compute_env_offsets(
            self.num_envs, self.env_offset, self.up_axis
        )
        self.env_offsets_wp = wp.array(
            self.env_offsets, dtype=wp.vec3, device=self.device
        )
        try:
            articulation_builder = wp.sim.ModelBuilder(
                up_vector=up_vector, gravity=self.gravity
            )
            self.create_articulation(articulation_builder)
            for i in trange(
                self.num_envs, desc=f"Creating {self.num_envs} environments"
            ):
                xform = wp.transform(self.env_offsets[i], wp.quat_identity())
                builder.add_builder(
                    articulation_builder,
                    xform,
                    separate_collision_group=self.separate_collision_group_per_env,
                )
            self.bodies_per_env = articulation_builder.body_count
            self.dof_q_per_env = articulation_builder.joint_coord_count
            self.dof_qd_per_env = articulation_builder.joint_dof_count
        except NotImplementedError:
            # custom simulation setup where something other than an articulation is used
            self.setup(builder)
            self.bodies_per_env = builder.body_count
            self.dof_q_per_env = builder.joint_coord_count
            self.dof_qd_per_env = builder.joint_dof_count

        self.model = builder.finalize(
            requires_grad=self.requires_grad, device=self.device
        )
        self.customize_model(self.model)
        self.device = self.model.device
        if not self.model.device.is_cuda:
            self.use_graph_capture = False
        self.model.ground = self.activate_ground_plane

        self.model.joint_attach_ke = self.joint_attach_ke
        self.model.joint_attach_kd = self.joint_attach_kd

        if self.integrator_type == IntegratorType.EULER:
            self.sim_substeps = self.sim_substeps_euler
            self.integrator = wp.sim.SemiImplicitIntegrator(**self.euler_settings)
        elif self.integrator_type == IntegratorType.XPBD:
            self.sim_substeps = self.sim_substeps_xpbd
            self.integrator = wp.sim.XPBDIntegrator(**self.xpbd_settings)
        elif self.integrator_type == IntegratorType.FEATHERSTONE:
            self.sim_substeps = self.sim_substeps_featherstone
            if self.featherstone_update_mass_matrix_once_per_step:
                self.featherstone_settings["update_mass_matrix_every"] = (
                    self.sim_substeps
                )
            self.integrator = wp.sim.FeatherstoneIntegrator(
                self.model, **self.featherstone_settings
            )

        if self.episode_frames is None:
            self.episode_frames = int(self.episode_duration / self.frame_dt)
        self.sim_dt = self.frame_dt / max(1, self.sim_substeps)
        self.sim_steps = self.episode_frames * self.sim_substeps
        self.sim_step = 0
        self.sim_time = 0.0
        self.invalidate_cuda_graph = False

        self.controls = []
        self.num_controls = self.sim_steps if self.requires_grad else 1
        for _ in range(self.num_controls):
            control = self.model.control()
            self.customize_control(control)
            self.controls.append(control)

        assert len(self.controllable_dofs) == len(self.control_gains)
        assert len(self.controllable_dofs) == len(self.control_limits)

        self.controllable_dofs_wp = wp.array(
            self.controllable_dofs, dtype=int, device=self.device
        )
        self.control_gains_wp = wp.array(
            self.control_gains, dtype=float, device=self.device
        )
        self.control_limits_wp = wp.array(
            self.control_limits, dtype=float, device=self.device
        )

        if self.requires_grad:
            self.states = []
            for _ in range(self.sim_steps + 1):
                state = self.model.state()
                self.customize_state(state)
                self.states.append(state)
            self.update = self.update_grad
        else:
            # set up current and next state to be used by the integrator
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.customize_state(self.state_0)
            self.customize_state(self.state_1)
            self.update = self.update_nograd
            if self.use_graph_capture:
                self.state_temp = self.model.state()
            else:
                self.state_temp = None

        self.renderer = None
        if self.profile:
            self.render_mode = RenderMode.NONE

        self._kinetic_energy = None
        self._potential_energy = None

        if setup_renderer:
            self.setup_renderer()

        self.extras = {}

    def setup_renderer(self):
        if self.render_mode == RenderMode.OPENGL:
            self.renderer = wp.sim.render.SimRendererOpenGL(
                self.model,
                self.sim_name,
                up_axis=self.up_axis,
                show_rigid_contact_points=self.show_rigid_contact_points,
                contact_points_radius=self.contact_points_radius,
                show_joints=self.show_joints,
                **self.opengl_render_settings,
            )
            if self.use_tiled_rendering:
                floor_id = self.model.shape_count - 1
                visible_shapes = self.model.shape_visible.numpy()
                num_visible_shapes = len(np.where(visible_shapes == 1)[0])
                instance_ids = np.arange(num_visible_shapes).tolist()
                # all shapes except the floor
                shapes_per_env = (num_visible_shapes - 1) // self.num_envs
                additional_instances = []
                if self.activate_ground_plane:
                    additional_instances.append(floor_id)
                view_matrices = self.opengl_tile_render_settings.get("view_matrices")
                if view_matrices is None and self.opengl_tile_render_settings.get(
                    "move_cam_pos_to_env_offsets", True
                ):
                    cam_pos = np.array(self.renderer._camera_pos)
                    cam_front = self.renderer._camera_front
                    cam_up = self.renderer._camera_up
                    view_matrices = []
                    for i in range(self.num_envs):
                        view_matrices.append(
                            self.renderer.compute_view_matrix(
                                cam_pos + self.env_offsets[i], cam_front, cam_up
                            )
                        )
                self.renderer.setup_tiled_rendering(
                    instances=[
                        instance_ids[i * shapes_per_env : (i + 1) * shapes_per_env]
                        + additional_instances
                        for i in range(self.num_envs)
                    ],
                    tile_width=self.opengl_tile_render_settings.get("tile_width"),
                    tile_height=self.opengl_tile_render_settings.get("tile_height"),
                    tile_ncols=self.opengl_tile_render_settings.get("tile_ncols"),
                    tile_nrows=self.opengl_tile_render_settings.get("tile_nrows"),
                    tile_positions=self.opengl_tile_render_settings.get(
                        "tile_positions"
                    ),
                    tile_sizes=self.opengl_tile_render_settings.get("tile_sizes"),
                    projection_matrices=self.opengl_tile_render_settings.get(
                        "projection_matrices"
                    ),
                    view_matrices=view_matrices,
                )
        elif self.render_mode == RenderMode.USD:
            extension = ".usda" if self.usd_save_as_usda else ".usd"
            filename = os.path.join(
                os.path.dirname(__file__), "..", "outputs", self.sim_name + extension
            )
            self.renderer = wp.sim.render.SimRendererUsd(
                self.model,
                filename,
                up_axis=self.up_axis,
                show_rigid_contact_points=self.show_rigid_contact_points,
                **self.usd_render_settings,
            )

    @property
    def uses_generalized_coordinates(self):
        # whether the model uses generalized or maximal coordinates (joint q/qd vs body q/qd) in the state
        return (
            self.integrator_type == IntegratorType.FEATHERSTONE
            or self.integrator_type == IntegratorType.NEURAL
        )

    def create_articulation(self, builder: wp.sim.ModelBuilder):
        raise NotImplementedError

    def setup(self, builder):
        pass

    def before_simulate(self):
        pass

    def after_simulate(self):
        pass

    def before_step(
        self,
        state: wp.sim.State,
        next_state: wp.sim.State,
        control: wp.sim.Control,
        eval_collisions: bool = True,
    ):
        pass

    def after_step(
        self,
        state: wp.sim.State,
        next_state: wp.sim.State,
        control: wp.sim.Control,
        eval_collisions: bool = True,
    ):
        pass

    def before_update(self):
        pass

    def after_update(self):
        pass

    def custom_render(self, render_state, renderer):
        pass

    @property
    def state(self) -> wp.sim.State:
        """
        Shortcut to the current state
        """
        if self.requires_grad:
            return self.states[self.sim_step]
        return self.state_0

    @property
    def next_state(self) -> wp.sim.State:
        """
        Shortcut to subsequent state (the state to which we will write to at the next substep)

        Note: This property is only rarely useful since it points to the state that will be written to
        at the next substep, which is not yet computed. If requires_grad is False, this property will
        point to the state before the last substep.
        """
        if self.requires_grad:
            return self.states[self.sim_step + 1]
        return self.state_1

    @property
    def control(self):
        return self.controls[
            min(len(self.controls) - 1, max(0, self.sim_step % self.sim_steps))
        ]

    def assign_control(
        self,
        actions: wp.array,
        control: wp.sim.Control,
        state: wp.sim.State,
    ):
        assert actions.ndim == 2
        assert actions.shape[0] == self.num_envs
        control_full_dim = len(self.control_input) // self.num_envs
        wp.launch(
            assign_controls,
            dim=self.num_envs,
            inputs=[
                actions,
                self.control_gains_wp,
                self.control_limits_wp,
                self.controllable_dofs_wp,
                control_full_dim,
            ],
            outputs=[
                control.joint_act,
            ],
            device=self.device,
        )

    def apply_pd_control(
        self,
        control_out: wp.array = None,
        body_f: wp.array = None,
        target_q: wp.array = None,
        target_qd: wp.array = None,
        target_ke: wp.array = None,
        target_kd: wp.array = None,
        joint_q: wp.array = None,
        joint_qd: wp.array = None,
        body_q: wp.array = None,
        apply_as_body_forces: bool = False,
    ):
        if self.model.joint_count == 0:
            return
        if apply_as_body_forces:
            if body_f is None:
                body_f = self.state.body_f
            if body_q is None:
                body_q = self.state.body_q
        if control_out is None:
            control_out = self.control_input
        if joint_q is None:
            joint_q = self.state.joint_q
        if joint_qd is None:
            joint_qd = self.state.joint_qd

        generate_pd_control(
            self.model,
            joint_torques=control_out,
            joint_q=joint_q,
            joint_qd=joint_qd,
            target_q=target_q,
            target_qd=target_qd,
            target_ke=target_ke,
            target_kd=target_kd,
        )
        if apply_as_body_forces:
            convert_joint_torques_to_body_forces(
                self.model,
                body_q=body_q,
                joint_torques=control_out,
                body_f=body_f,
            )
        return control_out

    @property
    def control_input(self):
        # points to the actuation input of the control
        return self.control.joint_act

    @property
    def joint_act(self):
        return self.control.joint_act

    @property
    def joint_act_dim(self):
        return len(self.control_input) // self.num_envs

    def customize_state(self, state: wp.sim.State):
        pass

    def customize_control(self, control: wp.sim.Control):
        pass

    def customize_model(self, model):
        pass

    def compute_cost_termination(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        step: int,
        max_episode_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        pass

    @property
    def control_dim(self):
        return len(self.controllable_dofs)

    @property
    def observation_dim(self):
        # default observation consists of generalized joint positions and velocities
        return self.dof_q_per_env + self.dof_qd_per_env

    def compute_observations(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        if not self.uses_generalized_coordinates:
            # evaluate generalized coordinates
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)

        assert self.num_envs == self.model.articulation_count
        wp.launch(
            assign_joint_q_qd_obs,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_type,
                self.env_offsets_wp,
            ],
            outputs=[observations],
            device=self.device,
        )

    def get_extras(self, extras: dict):
        for k, v in self.extras.items():
            if isinstance(v, dict):
                if k not in extras:
                    extras[k] = {}
                for k2, v2 in v.items():
                    extras[k][k2] = v2
            else:
                extras[k] = v

    def step(
        self,
        state: wp.sim.State,
        next_state: wp.sim.State,
        control: wp.sim.Control,
        eval_collisions: bool = True,
    ):
        self.extras = {}
        state.clear_forces()
        self.before_step(
            state=state,
            next_state=next_state,
            control=control,
            eval_collisions=eval_collisions,
        )
        if self.custom_dynamics is not None:
            self.custom_dynamics(self.model, state, next_state, self.sim_dt, control)
        else:
            if eval_collisions and not getattr(
                self.model, "separate_ground_contacts", False
            ):
                with wp.ScopedTimer(
                    "collision_handling", color="orange", active=self.enable_timers
                ):
                    wp.sim.collide(
                        self.model,
                        state,
                        edge_sdf_iter=self.edge_sdf_iter,
                        iterate_mesh_vertices=self.rigid_contact_iterate_mesh_vertices,
                    )
            with wp.ScopedTimer("simulation", color="red", active=self.enable_timers):
                self.integrator.simulate(
                    self.model, state, next_state, self.sim_dt, control
                )
        self.after_step(state, next_state, control, eval_collisions=eval_collisions)
        self.sim_time += self.sim_dt
        self.sim_step += 1

    def update_nograd(self):
        self.before_update()
        if self.use_graph_capture:
            state_0_dict = self.state_0.__dict__
            state_1_dict = self.state_1.__dict__
            state_temp_dict = (
                self.state_temp.__dict__ if self.state_temp is not None else None
            )
        for i in range(self.sim_substeps):
            if self.handle_collisions_once_per_step and i > 0:
                eval_collisions = False
            else:
                eval_collisions = True
            self.step(
                self.state_0,
                self.state_1,
                self.control,
                eval_collisions=eval_collisions,
            )
            if i < self.sim_substeps - 1 or not self.use_graph_capture:
                # we can just swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0
            elif self.use_graph_capture:
                assert (
                    hasattr(self, "state_temp") and self.state_temp is not None
                ), "state_temp must be allocated when using graph capture"
                # swap states by actually copying the state arrays to make sure the graph capture works
                for key, value in state_0_dict.items():
                    if isinstance(value, wp.array):
                        if key not in state_temp_dict:
                            state_temp_dict[key] = wp.empty_like(value)
                        state_temp_dict[key].assign(value)
                        state_0_dict[key].assign(state_1_dict[key])
                        state_1_dict[key].assign(state_temp_dict[key])
        self.after_update()
        return self.state_0

    def update_grad(self):
        self.before_update()
        for i in range(self.sim_substeps):
            if self.handle_collisions_once_per_step and i > 0:
                eval_collisions = False
            else:
                eval_collisions = True
            self.step(
                self.states[self.sim_step],
                self.states[self.sim_step + 1],
                self.controls[self.sim_step],
                eval_collisions=eval_collisions,
            )
        self.after_update()
        return self.states[self.sim_step]

    def render(self, state=None):
        if self.renderer is not None:
            with wp.ScopedTimer("render", color="yellow", active=self.enable_timers):
                self.renderer.begin_frame(self.sim_time)
                if self.requires_grad:
                    # ensure we do not render beyond the last state
                    render_state = (
                        state or self.states[min(self.sim_steps, self.sim_step)]
                    )
                else:
                    render_state = state or self.state
                if self.uses_generalized_coordinates:
                    wp.sim.eval_fk(
                        self.model,
                        render_state.joint_q,
                        render_state.joint_qd,
                        None,
                        render_state,
                    )
                with wp.ScopedTimer(
                    "custom_render", color="orange", active=self.enable_timers
                ):
                    self.custom_render(render_state, renderer=self.renderer)
                self.renderer.render(render_state)
                self.renderer.end_frame()

    def reset(self):
        self.sim_time = 0.0
        self.sim_step = 0

        if self.eval_fk:
            wp.sim.eval_fk(
                self.model, self.model.joint_q, self.model.joint_qd, None, self.state
            )
            self.model.body_q.assign(self.state.body_q)
            self.model.body_qd.assign(self.state.body_qd)

        if self.model.particle_count > 1:
            self.model.particle_grid.build(
                self.state.particle_q,
                self.model.particle_max_radius * 2.0,
            )

        self.reset_envs()

    def reset_envs(self, env_ids: wp.array = None, state=None):
        """Reset environments where env_ids buffer indicates True. Resets all envs if env_ids is None."""
        if state is None:
            state = self.state
        if self.uses_generalized_coordinates:
            wp.launch(
                reset_generalized_coords,
                dim=self.num_envs,
                inputs=[
                    env_ids,
                    self.model.joint_q,
                    self.model.joint_qd,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                ],
                outputs=[
                    state.joint_q,
                    state.joint_qd,
                ],
                device=self.device,
                record_tape=False,
            )
        else:
            wp.launch(
                reset_maximal_coords,
                dim=self.num_envs,
                inputs=[
                    env_ids,
                    self.model.body_q,
                    self.model.body_qd,
                    self.bodies_per_env,
                ],
                outputs=[
                    state.body_q,
                    state.body_qd,
                ],
                device=self.device,
                record_tape=False,
            )
            # if env_ids is not None:
            #     print(env_ids.numpy().any())

    def after_reset(self):
        pass

    def run(self, opengl_render_start_paused=True, skip_save_render=False):
        # ---------------
        # run simulation
        if self.render_mode != RenderMode.NONE and self.renderer is None:
            self.setup_renderer()
        self.reset()

        self.before_simulate()

        if self.renderer is not None:
            self.render(self.state)

            if self.render_mode == RenderMode.OPENGL:
                self.renderer.paused = opengl_render_start_paused

        profiler = {}

        if self.use_graph_capture:
            with wp.ScopedCapture() as capture:
                self.update()
            graph = capture.graph

        if self.plot_body_coords:
            q_history = []
            q_history.append(self.state.body_q.numpy().copy())
            qd_history = []
            qd_history.append(self.state.body_qd.numpy().copy())
            delta_history = []
            delta_history.append(self.state.body_deltas.numpy().copy())
            num_con_history = []
            num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())
            self.use_graph_capture = False
        if self.plot_joint_coords or self.plot_joint_coords_qd:
            joint_q_history = []
            joint_qd_history = []
            if not self.uses_generalized_coordinates:
                joint_q = wp.zeros_like(self.model.joint_q)
                joint_qd = wp.zeros_like(self.model.joint_qd)
            self.use_graph_capture = False

        if not self.continuous_opengl_render or self.render_mode != RenderMode.OPENGL:
            frames = trange(self.episode_frames, desc=f"Running {self.sim_name}")
        else:
            frames = range(self.episode_frames)

        # simulate
        with wp.ScopedTimer(
            "run_loop",
            detailed=False,
            print=False,
            active=self.enable_timers or self.profile,
            dict=profiler,
        ):
            running = True
            while running:
                for f in frames:
                    if self.model.particle_count > 1:
                        self.model.particle_grid.build(
                            self.state.particle_q,
                            self.model.particle_max_radius * 2.0,
                        )
                    if self.use_graph_capture:
                        with wp.ScopedTimer(
                            "simulation_graph", color="green", active=self.enable_timers
                        ):
                            wp.capture_launch(graph)
                        self.sim_time += self.frame_dt
                        self.sim_step += self.sim_substeps
                    elif not self.requires_grad or self.sim_step < self.sim_steps:
                        with wp.ScopedTimer(
                            "update", color="green", active=self.enable_timers
                        ):
                            self.update()

                        if not self.profile:
                            if self.plot_body_coords:
                                q_history.append(self.state.body_q.numpy().copy())
                                qd_history.append(self.state.body_qd.numpy().copy())
                                delta_history.append(
                                    self.state.body_deltas.numpy().copy()
                                )
                                num_con_history.append(
                                    self.model.rigid_contact_inv_weight.numpy().copy()
                                )

                            if self.plot_joint_coords or self.plot_joint_coords_qd:
                                if self.uses_generalized_coordinates:
                                    joint_q = self.state.joint_q
                                    joint_qd = self.state.joint_qd
                                else:
                                    wp.sim.eval_ik(
                                        self.model, self.state, joint_q, joint_qd
                                    )
                                joint_q_history.append(joint_q.numpy().copy())
                                joint_qd_history.append(joint_qd.numpy().copy())

                    self.render()
                    if self.render_mode == RenderMode.OPENGL and self.renderer.has_exit:
                        running = False
                        break

                if (
                    not self.continuous_opengl_render
                    or self.render_mode != RenderMode.OPENGL
                ):
                    break

            wp.synchronize()

        self.after_simulate()

        if len(profiler["run_loop"]) > 0:
            avg_time = np.array(profiler["run_loop"]).mean() / self.episode_frames
            avg_steps_second = 1000.0 * float(self.num_envs) / avg_time
            fps = 1000.0 * float(self.num_envs) / avg_time

            print(
                f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}"
            )
        else:
            fps = None

        if self.renderer is not None and not skip_save_render:
            self.renderer.save()

        if self.plot_body_coords:
            import matplotlib.pyplot as plt

            q_history = np.array(q_history)
            qd_history = np.array(qd_history)
            delta_history = np.array(delta_history)
            num_con_history = np.array(num_con_history)

            # find bodies with non-zero mass
            body_indices = np.where(self.model.body_mass.numpy() > 0)[0]
            body_indices = body_indices[:5]  # limit number of bodies to plot

            fig, ax = plt.subplots(
                len(body_indices), 7, figsize=(10, 10), squeeze=False
            )
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            for i, j in enumerate(body_indices):
                ax[i, 0].set_title(f"Body {j} Position")
                ax[i, 0].grid()
                ax[i, 1].set_title(f"Body {j} Orientation")
                ax[i, 1].grid()
                ax[i, 2].set_title(f"Body {j} Linear Velocity")
                ax[i, 2].grid()
                ax[i, 3].set_title(f"Body {j} Angular Velocity")
                ax[i, 3].grid()
                ax[i, 4].set_title(f"Body {j} Linear Delta")
                ax[i, 4].grid()
                ax[i, 5].set_title(f"Body {j} Angular Delta")
                ax[i, 5].grid()
                ax[i, 6].set_title(f"Body {j} Num Contacts")
                ax[i, 6].grid()
                ax[i, 0].plot(q_history[:, j, :3])
                ax[i, 1].plot(q_history[:, j, 3:])
                ax[i, 2].plot(qd_history[:, j, 3:])
                ax[i, 3].plot(qd_history[:, j, :3])
                ax[i, 4].plot(delta_history[:, j, 3:])
                ax[i, 5].plot(delta_history[:, j, :3])
                ax[i, 6].plot(num_con_history[:, j])
                ax[i, 0].set_xlim(0, self.sim_steps)
                ax[i, 1].set_xlim(0, self.sim_steps)
                ax[i, 2].set_xlim(0, self.sim_steps)
                ax[i, 3].set_xlim(0, self.sim_steps)
                ax[i, 4].set_xlim(0, self.sim_steps)
                ax[i, 5].set_xlim(0, self.sim_steps)
                ax[i, 6].set_xlim(0, self.sim_steps)
                ax[i, 6].yaxis.get_major_locator().set_params(integer=True)
            plt.show()

        if (
            self.plot_joint_coords
            and len(joint_q_history) > 0
            or self.plot_joint_coords_qd
            and len(joint_qd_history) > 0
        ):
            import matplotlib.pyplot as plt

            if self.plot_joint_coords_qd:
                history = np.array(joint_qd_history)
            else:
                history = np.array(joint_q_history)
            dof = history.shape[1]
            ncols = int(np.ceil(np.sqrt(dof)))
            nrows = int(np.ceil(dof / float(ncols)))
            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                constrained_layout=True,
                figsize=(ncols * 3.5, nrows * 3.5),
                squeeze=False,
                sharex=True,
            )

            joint_id = 0
            joint_type_names = {
                wp.sim.JOINT_BALL: "ball",
                wp.sim.JOINT_REVOLUTE: "hinge",
                wp.sim.JOINT_PRISMATIC: "slide",
                wp.sim.JOINT_UNIVERSAL: "universal",
                wp.sim.JOINT_COMPOUND: "compound",
                wp.sim.JOINT_FREE: "free",
                wp.sim.JOINT_FIXED: "fixed",
                wp.sim.JOINT_DISTANCE: "distance",
                wp.sim.JOINT_D6: "D6",
            }
            joint_lower = self.model.joint_limit_lower.numpy()
            joint_upper = self.model.joint_limit_upper.numpy()
            joint_type = self.model.joint_type.numpy()
            while (
                joint_id < len(joint_type) - 1
                and joint_type[joint_id] == wp.sim.JOINT_FIXED
            ):
                # skip fixed joints
                joint_id += 1
            q_start = self.model.joint_q_start.numpy()
            qd_start = self.model.joint_qd_start.numpy()
            axis_start = self.model.joint_axis_start.numpy()
            qd_i = qd_start[joint_id]
            axis_i = axis_start[joint_id]
            for dim in range(ncols * nrows):
                ax = axes[dim // ncols, dim % ncols]
                if dim >= dof:
                    ax.axis("off")
                    continue
                ax.grid()
                ax.plot(history[:, dim])
                if joint_type[joint_id] != wp.sim.JOINT_FREE:
                    if not self.plot_joint_coords_qd:
                        lower = joint_lower[axis_i]
                        if abs(lower) < 2 * np.pi:
                            ax.axhline(lower, color="red")
                        upper = joint_upper[axis_i]
                        if abs(upper) < 2 * np.pi:
                            ax.axhline(upper, color="red")
                joint_name = joint_type_names[joint_type[joint_id]]
                if self.plot_joint_coords_qd:
                    ax.set_title(
                        f"$\\mathbf{{\\dot{{q}}_{{{dim}}}}}$ ({self.model.joint_name[joint_id]} / {joint_name} {joint_id})"
                    )
                else:
                    ax.set_title(
                        f"$\\mathbf{{q_{{{dim}}}}}$ ({self.model.joint_name[joint_id]} / {joint_name} {joint_id})"
                    )
                if (
                    joint_id < self.model.joint_count - 1
                    and q_start[joint_id + 1] == dim + 1
                ):
                    joint_id += 1
                    qd_i = qd_start[joint_id]
                    axis_i = axis_start[joint_id]
                else:
                    qd_i += 1
            plt.tight_layout()
            plt.show()

        return fps

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def compute_kinetic_energy(
        self,
        compute_body_v_s=True,
        compute_mass_matrix=True,
    ):
        assert (
            self.integrator_type == IntegratorType.FEATHERSTONE
        ), "Kinetic energy is only supported for Featherstone"
        if self._kinetic_energy is None:
            self._kinetic_energy = wp.empty(
                self.num_envs, dtype=float, device=self.device
            )
        if self.state.requires_grad:
            state_aug = self.state
        else:
            state_aug = self.integrator
        if not getattr(state_aug, "_featherstone_augmented", False):
            self.integrator.allocate_state_aux_vars(
                self.model, state_aug, self.state.requires_grad
            )
        wp.launch(
            eval_rigid_fk,
            dim=self.model.articulation_count,
            inputs=[
                self.model.articulation_start,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_q_start,
                self.state.joint_q,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.integrator.body_X_com,
                self.model.joint_axis,
                self.model.joint_axis_start,
                self.model.joint_axis_dim,
            ],
            outputs=[self.state.body_q, state_aug.body_q_com],
            device=self.device,
        )

        if compute_body_v_s:
            wp.launch(
                eval_rigid_id,
                dim=self.model.articulation_count,
                inputs=[
                    self.model.articulation_start,
                    self.model.joint_type,
                    self.model.joint_parent,
                    self.model.joint_child,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.state.joint_q,
                    self.state.joint_qd,
                    self.model.joint_axis,
                    self.model.joint_axis_start,
                    self.model.joint_axis_dim,
                    self.integrator.body_I_m,
                    self.state.body_q,
                    self.integrator.body_q_com,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                    self.model.gravity,
                ],
                outputs=[
                    self.integrator.joint_S_s,
                    self.integrator.body_I_s,
                    state_aug.body_v_s,
                    self.integrator.body_f_s,
                    self.integrator.body_a_s,
                ],
                device=self.device,
            )

        if compute_mass_matrix:
            wp.launch(
                eval_rigid_mass,
                dim=self.model.articulation_count,
                inputs=[
                    self.model.articulation_start,
                    self.integrator.articulation_M_start,
                    state_aug.body_I_s,
                ],
                outputs=[self.integrator.M],
                device=self.device,
            )

        wp.launch(
            eval_kinetic_energy,
            dim=self.num_envs,
            inputs=[
                self.model.articulation_start,
                self.integrator.articulation_M_start,
                state_aug.body_v_s,
                self.integrator.M,
            ],
            outputs=[self._kinetic_energy],
            device=self.device,
        )

        return self._kinetic_energy

    def compute_potential_energy(self, eval_fk_if_generalized=True):
        if self._potential_energy is None:
            self._potential_energy = wp.empty(
                self.num_envs, dtype=float, device=self.device
            )
        if eval_fk_if_generalized and self.uses_generalized_coordinates:
            wp.sim.eval_fk(
                self.model, self.state.joint_q, self.state.joint_qd, None, self.state
            )

        wp.launch(
            eval_potential_energy,
            dim=self.num_envs,
            inputs=[
                self.state.body_q,
                self.model.body_com,
                self.model.body_mass,
                self.model.gravity,
                self.bodies_per_env,
            ],
            outputs=[self._potential_energy],
            device=self.device,
        )
        return self._potential_energy


def run_env(Environment, opengl_render_start_paused=True, skip_save_render=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integrator",
        help="Type of integrator",
        type=IntegratorType,
        choices=list(IntegratorType),
        default=Environment.integrator_type.value,
    )
    parser.add_argument(
        "--visualizer",
        help="Type of renderer",
        type=RenderMode,
        choices=list(RenderMode),
        default=Environment.render_mode.value,
    )
    parser.add_argument(
        "--num_envs",
        help="Number of environments to simulate",
        type=int,
        default=Environment.num_envs,
    )
    parser.add_argument("--profile", help="Enable profiling", type=bool, default=False)
    args = parser.parse_args()
    env_args = dict(
        integrator_type=args.integrator,
        render_mode=args.visualizer,
        profile=args.profile,
    )

    if args.profile:
        import matplotlib.pyplot as plt

        env_count = 2
        env_times = []
        env_size = []

        for i in range(15):
            demo = Environment(
                num_envs=env_count,
                **env_args,
            )
            steps_per_second = demo.run()

            env_size.append(env_count)
            env_times.append(steps_per_second)

            env_count *= 2

        # dump times
        for i in range(len(env_times)):
            print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

        # plot
        plt.figure(1)
        plt.plot(env_size, env_times)
        plt.xscale("log")
        plt.xlabel("Number of Envs")
        plt.yscale("log")
        plt.ylabel("Steps/Second")
        plt.show()
    else:
        demo = Environment(
            num_envs=args.num_envs,
            **env_args,
        )
        return demo.run(
            opengl_render_start_paused=opengl_render_start_paused,
            skip_save_render=skip_save_render,
        )
